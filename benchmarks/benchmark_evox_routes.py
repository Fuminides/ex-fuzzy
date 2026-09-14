"""Complete classification fits on each population evaluation route, including EvoX GPU.

Routes:

``pymoo``
    The default PyMoo backend with the current CPU evaluator.  It runs a
    different genetic algorithm from EvoX, so it is timed as the CPU reference
    and never compared for identical results.
``individual``
    EvoX without the fit-local scope: one ``FitRuleBase._evaluate`` call per
    candidate, which is how EvoX classification was evaluated before it shared
    PyMoo's caches.
``cpu``
    EvoX with the fit-local caches, within-population deduplication and the
    measured scalar/batched CPU choice, with the device objective disabled.
``device``
    The same plus the exact PyTorch objective on the EvoX device.  On a CUDA
    machine that is the default behavior; on a CPU-only machine CPU tensors are
    enabled explicitly, which measures overhead rather than a speedup.

Data and model settings match ``benchmark_speedup.py``'s scaling campaign:
``make_classification`` with ``features - 1`` informative features, three
classes and data seed 42, fixed partitions from ``construct_partitions``,
Type-1 sets, 20 rules, four antecedents and tolerance 0.

Timings are withheld unless every EvoX route produced the identical
best-fitness history, final population, final fitness and predictions for each
seed.  The device route's verification outcome, the CPU and device seconds it
measured, and which generations the device scored are recorded as evidence.
When the ``individual`` route runs, the fitness cache capacity is also replayed
offline over the populations it evaluated.

EvoX 1.4.0 cannot be imported with PyTorch 2.6 (a custom-op schema error in its
package initialization).  Install EvoX 1.3.0, whose SBX and polynomial mutation
are identical, or pass ``--evox-source`` to load those operators straight from
an EvoX source tree.  The EvoX routes never import pymoo; only the ``pymoo``
route needs it.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import socket
import sys
import time
import types
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ex_fuzzy"))

import numpy as np  # noqa: E402

EVOX_ROUTES = ("individual", "cpu", "device")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_evox_operators(source: Path) -> None:
    """Expose EvoX's SBX and polynomial mutation without its package init."""
    package = source / "evox"
    evox = types.ModuleType("evox")
    sys.modules["evox"] = evox
    evox.utils = _load_module(package / "utils" / "jit_fix_operator.py", "evox.utils")
    sys.modules["evox.utils"] = evox.utils
    operators = types.ModuleType("evox.operators")
    operators.crossover = types.SimpleNamespace(simulated_binary=_load_module(
        package / "operators" / "crossover" / "sbx.py", "evox_sbx").simulated_binary)
    operators.mutation = types.SimpleNamespace(polynomial_mutation=_load_module(
        package / "operators" / "mutation" / "pm_mutation.py", "evox_pm").polynomial_mutation)
    evox.operators = operators
    sys.modules["evox.operators"] = operators


def _replay_cache(populations: list, capacity) -> int:
    """Genotypes a deduplicating LRU cache of ``capacity`` entries would score."""
    cache, scored = OrderedDict(), 0
    for genes in populations:
        fresh = OrderedDict()
        for gene in genes:
            key = gene.tobytes()
            if key in cache:
                cache.move_to_end(key)
            elif key not in fresh:
                fresh[key] = True
                scored += 1
        for key in fresh:
            cache[key] = True
            while capacity is not None and len(cache) > capacity:
                cache.popitem(last=False)
    return scored


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _version(package: str):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, nargs="+", default=[150, 1000])
    parser.add_argument("--features", type=int, nargs="+", default=[10])
    parser.add_argument("--partitions", nargs="+", choices=("fixed", "optimized"),
                        default=["fixed", "optimized"])
    parser.add_argument("--rules", type=int, default=20)
    parser.add_argument("--ants", type=int, default=4)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--routes", nargs="+", choices=("pymoo",) + EVOX_ROUTES,
                        default=list(EVOX_ROUTES))
    parser.add_argument("--evox-source", type=Path,
                        help="EvoX source tree to load operators from if 'import evox' fails")
    parser.add_argument("--json", type=Path, help="write the results to this file")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import evox.operators  # noqa: F401
    except Exception:
        if args.evox_source is None:
            raise SystemExit("EvoX cannot be imported; pass --evox-source <EvoX source tree>.")
        _install_evox_operators(args.evox_source)

    import torch
    from sklearn.datasets import make_classification

    import ex_fuzzy._fitness as fitness_module
    import ex_fuzzy._torch_fitness as torchfit
    import ex_fuzzy.evolutionary_fit as evf
    import ex_fuzzy.fuzzy_sets as fs
    import ex_fuzzy.utils as utils

    cuda = torch.cuda.is_available()
    if cuda:
        # Context creation is not fitness work; keep it out of every timer.
        torch.ones(1, device="cuda").sum().item()
    device_types = ("cuda",) if cuda else ("cpu",)

    recorded, verifications, generations = [], [], []
    original_population = evf.FitRuleBase._evaluate_gene_population
    original_route = evf.FitRuleBase._score_on_best_route
    original_verify = torchfit.DeviceRoute.verify

    def recording_population(self, genes, device=None):
        recorded.append(np.array(genes))
        return original_population(self, genes, device)

    def recording_route(self, fresh, population, device=None):
        start = time.perf_counter()
        fitness, on_device = original_route(self, fresh, population, device)
        generations.append({"candidates": int(len(fresh)), "on_device": bool(on_device),
                            "seconds": time.perf_counter() - start})
        return fitness, on_device

    def recording_verify(self, expected, actual, cpu_seconds=None, device_seconds=None):
        verified = original_verify(self, expected, actual, cpu_seconds, device_seconds)
        verifications.append({"verified": verified, "candidates": int(len(expected)),
                              "cpu_seconds": cpu_seconds, "device_seconds": device_seconds,
                              "settled_on_device": self.probe.decision == self.DEVICE})
        return verified

    def unscoped(problem, enabled, population=None):
        return nullcontext()

    def route_context(route):
        if route == "individual":
            return patch.object(fitness_module, "_fitness_cache_scope", unscoped)
        if route == "cpu":
            return patch.object(evf.FitRuleBase, "torch_devices", ())
        if route == "device":
            return patch.object(evf.FitRuleBase, "torch_devices", device_types)
        return nullcontext()

    def run(route, X, y, partitions, seed):
        recorded.clear()
        verifications.clear()
        generations.clear()
        model = evf.BaseFuzzyRulesClassifier(
            nRules=args.rules, nAnts=args.ants, fuzzy_type=fs.FUZZY_SETS.t1,
            backend="pymoo" if route == "pymoo" else "evox",
            linguistic_variables=partitions, ds_mode=0, tolerance=0.0, allow_unknown=False)
        with route_context(route), \
                patch.object(evf.FitRuleBase, "_evaluate_gene_population", recording_population), \
                patch.object(evf.FitRuleBase, "_score_on_best_route", recording_route), \
                patch.object(torchfit.DeviceRoute, "verify", recording_verify):
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.fit(X, y, n_gen=args.generations, pop_size=args.population,
                      random_state=seed, patience=None)
            if cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        result = model.optimization_result_
        record = {"seconds": seconds, "performance": float(model.performance)}
        outcome = None
        if route != "pymoo":
            outcome = (np.asarray(result["history"]["best_fitness"]), result["pop"],
                       result["fitness"], model.predict(X))
            record.update(gpu_accelerated=bool(result["gpu_accelerated"]),
                          verifications=list(verifications),
                          device_generations=sum(g["on_device"] for g in generations),
                          scored_generations=len(generations))
        return record, outcome, list(recorded)

    report = {
        "host": socket.gethostname(), "cpu": _cpu_model(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "python": platform.python_version(), "torch": torch.__version__,
        "packages": {name: _version(name) for name in ("numpy", "pymoo", "scikit-learn")},
        "threads": {key: os.environ.get(key) for key in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "workloads": [],
    }
    print(f"host={report['host']} cpu={report['cpu']} gpu={report['gpu']} "
          f"torch={report['torch']} numpy={report['packages']['numpy']}", flush=True)
    order = random.Random(11)
    for samples in args.samples:
        for features in args.features:
            X, y = make_classification(n_samples=samples, n_features=features,
                                       n_informative=features - 1, n_redundant=0,
                                       n_classes=3, random_state=42)
            for mode in args.partitions:
                partitions = (utils.construct_partitions(X, fs.FUZZY_SETS.t1)
                              if mode == "fixed" else None)
                for seed in args.seeds:
                    runs = {route: [] for route in args.routes}
                    outcomes, populations = {}, None
                    for _ in range(args.repeats):
                        routes = list(args.routes)
                        order.shuffle(routes)
                        for route in routes:
                            print(f"n={samples} features={features} {mode} seed={seed} {route}",
                                  flush=True)
                            record, outcome, pops = run(route, X, y, partitions, seed)
                            runs[route].append(record)
                            if outcome is not None:
                                outcomes.setdefault(route, outcome)
                            if route == "individual" and populations is None:
                                populations = pops
                            print(f"  {record['seconds']:.3f} s", flush=True)
                    reference = next(iter(outcomes.values()), None)
                    identical = all(
                        all(np.array_equal(a, b) for a, b in zip(reference, outcome))
                        for outcome in outcomes.values())
                    entry = {"samples": samples, "features": features, "partitions": mode,
                             "seed": seed, "generations": args.generations,
                             "population": args.population, "evox_identical": identical,
                             "runs": runs if identical else {
                                 route: [{k: v for k, v in r.items() if k != "seconds"}
                                         for r in records]
                                 for route, records in runs.items()}}
                    print(f"== samples={samples} features={features} {mode} seed={seed} "
                          f"evox_identical={identical}")
                    if identical:
                        entry["median_seconds"] = {
                            route: float(np.median([r["seconds"] for r in records]))
                            for route, records in runs.items()}
                        for route, seconds in entry["median_seconds"].items():
                            print(f"  {route:10s} median {seconds:9.3f} s")
                    else:
                        print("  timings withheld: the EvoX routes produced different searches")
                    if populations is not None:
                        capacities = {"256": 256, "1x": args.population,
                                      "2x": 2 * args.population, "4x": 4 * args.population,
                                      "8x": 8 * args.population, "unbounded": None}
                        entry["evaluations"] = int(sum(len(p) for p in populations))
                        entry["scored_by_cache_capacity"] = {
                            name: _replay_cache(populations, capacity)
                            for name, capacity in capacities.items()}
                        print("  genotypes scored by cache capacity: " + ", ".join(
                            f"{k}={v}" for k, v in entry["scored_by_cache_capacity"].items()))
                    report["workloads"].append(entry)
                    if args.json is not None:
                        args.json.parent.mkdir(parents=True, exist_ok=True)
                        temporary = args.json.with_suffix(".tmp")
                        temporary.write_text(json.dumps(report, indent=2))
                        temporary.replace(args.json)


if __name__ == "__main__":
    main()
