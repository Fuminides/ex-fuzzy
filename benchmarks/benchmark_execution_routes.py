"""Opt-in diagnostics for CPU execution routes (C04/C05/C08/C10).

Run only on an otherwise idle machine, for example:

    python benchmarks/benchmark_execution_routes.py --samples 1000 --generations 10
    python benchmarks/benchmark_execution_routes.py --independent-fits

This is a measurement tool, not a performance test.  It keeps seeds and search
budgets fixed, verifies exact fitted outcomes, and reports payload bytes
separately from timings.  Payload bytes are *not* peak RSS: they exclude Python
objects, allocator overhead, temporary arrays, worker copies, and process state.
"""
import argparse
from contextlib import nullcontext
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import pymoo
from sklearn.datasets import make_classification

# Always exercise this checkout rather than an installed release.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils

fitness_module = importlib.import_module(evf.__package__ + '._fitness')


def _make_data(args):
    return make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=max(2, args.features - 1),
        n_redundant=0,
        n_classes=args.classes,
        random_state=args.data_seed,
    )


def _model(args, runner):
    return evf.BaseFuzzyRulesClassifier(
        nRules=args.rules, nAnts=min(args.antecedents, args.features),
        fuzzy_type=fs.FUZZY_SETS.t1, runner=runner,
    )


def _fingerprint(model, X):
    """Search-visible state that must agree across an exact execution route."""
    result = model.optimization_result_
    return {
        'selected_x': np.asarray(result['X']).copy(),
        'predictions': np.asarray(model.predict(X)).copy(),
        'scores': np.asarray([rule.score for rule in model.rule_base.get_rules()]),
        'n_eval': int(result['algorithm'].evaluator.n_eval),
    }


def _assert_fit_equal(reference, candidate, label):
    np.testing.assert_array_equal(candidate['selected_x'], reference['selected_x'],
                                  err_msg=f'{label}: selected chromosome differs')
    np.testing.assert_array_equal(candidate['predictions'], reference['predictions'],
                                  err_msg=f'{label}: predictions differ')
    np.testing.assert_array_equal(candidate['scores'], reference['scores'],
                                  err_msg=f'{label}: retained rule scores differ')
    if candidate['n_eval'] != reference['n_eval']:
        raise AssertionError(f"{label}: n_eval {candidate['n_eval']} != {reference['n_eval']}")


def _fit_once(args, X, y, runner, cache_enabled):
    model = _model(args, runner)
    scope = nullcontext()
    if not cache_enabled:
        # Diagnostic-only switch: preserve the production objective but replace
        # the private fit-local memoization context with an inert context.
        scope = patch.object(fitness_module, '_fitness_cache_scope',
                             lambda *unused: nullcontext())
    with scope:
        start = time.perf_counter()
        model.fit(X, y, n_gen=args.generations, pop_size=args.population,
                  random_state=args.seed, patience=None)
        elapsed = time.perf_counter() - start
    return elapsed, _fingerprint(model, X)


def fit_route_diagnostic(args, X, y):
    """Compare complete serial and threaded fits, including B02's serial path."""
    routes = [
        ('serial_cached', 1, True),
        ('serial_uncached', 1, False),
        ('threads_2', 2, True),
        ('threads_4', 4, True),
    ]
    results = {}
    reference = None
    for name, runner, cache_enabled in routes:
        elapsed, fingerprint = _fit_once(args, X, y, runner, cache_enabled)
        if reference is None:
            reference = fingerprint
        else:
            _assert_fit_equal(reference, fingerprint, name)
        results[name] = {
            'seconds': elapsed,
            'runner': runner,
            'fitness_cache_requested': cache_enabled,
            # Runner paths intentionally bypass B02 even though the benchmark
            # argument says True: this describes production behavior.
            'fitness_cache_active': cache_enabled and runner == 1,
            'n_eval': fingerprint['n_eval'],
        }

    cached = results['serial_cached']['seconds']
    uncached = results['serial_uncached']['seconds']
    return {
        'routes': results,
        'b02_serial_cache_speedup_uncached_over_cached': uncached / cached if cached else None,
        'note': ('B02 is deliberately active only for the serial built-in PyMoo fit; '
                 'worker routes are compared against serial_cached for exact search parity '
                 'but do not use its memoization.'),
    }


# Spawn workers import this module and initialize one persistent problem copy.
_PROCESS_PROBLEM = None


def _init_process_problem(problem_payload):
    global _PROCESS_PROBLEM
    _PROCESS_PROBLEM = pickle.loads(problem_payload)


def _process_evaluate(gene):
    out = {}
    _PROCESS_PROBLEM._evaluate(np.asarray(gene), out)
    return float(np.asarray(out['F']).reshape(-1)[0])


def _candidate_problem(args, X, y):
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    return evf.FitRuleBase(
        X, y, nRules=args.rules, nAnts=min(args.antecedents, args.features),
        n_classes=args.classes, linguistic_variables=partitions,
        fuzzy_type=fs.FUZZY_SETS.t1,
    )


def _evaluate_serial(problem, genes):
    start = time.perf_counter()
    values = []
    for gene in genes:
        out = {}
        problem._evaluate(gene, out)
        values.append(float(np.asarray(out['F']).reshape(-1)[0]))
    return time.perf_counter() - start, np.asarray(values)


def _payload_bytes(X, y, problem):
    memberships = np.asarray(problem._precomputed_truth)
    return {
        'X': int(np.asarray(X).nbytes),
        'y': int(np.asarray(y).nbytes),
        'precomputed_memberships': int(memberships.nbytes),
        'sum': int(np.asarray(X).nbytes + np.asarray(y).nbytes + memberships.nbytes),
    }


def process_candidate_diagnostic(args, X, y):
    """Measure spawned persistent-worker overhead without sharing mutable state."""
    problem = _candidate_problem(args, X, y)
    rng = np.random.default_rng(args.seed)
    genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                         size=(args.candidates, problem.n_var), dtype=np.int64)
    serial_seconds, serial_values = _evaluate_serial(problem, genes)

    memory = _payload_bytes(X, y, problem)
    serialize_start = time.perf_counter()
    try:
        payload = pickle.dumps(problem, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        # This route must describe unsupported serialization honestly, rather
        # than repairing or silently changing the problem representation.
        return {
            'status': 'blocked',
            'serial_candidate_seconds': serial_seconds,
            'candidate_count': len(genes),
            'serialization_seconds_before_failure': time.perf_counter() - serialize_start,
            'serialization_error_type': type(error).__name__,
            'serialization_error': str(error),
            'payload_array_bytes_not_rss': memory,
            'memory_note': ('Array payload bytes are not RSS or a complete worker-memory bound: '
                            'they exclude Python objects, pickle buffers, temporary arrays, '
                            'allocator overhead, and per-process runtime state.'),
            'blocker_note': ('Spawn ProcessPool cannot be measured until FitRuleBase and its '
                             'referenced fuzzy-set state are picklable. This benchmark does not '
                             'repair that library limitation.'),
        }
    serialization_seconds = time.perf_counter() - serialize_start
    context = mp.get_context('spawn')
    pool_start = time.perf_counter()
    with context.Pool(args.processes, initializer=_init_process_problem,
                      initargs=(payload,)) as pool:
        first_values = np.asarray(pool.map(_process_evaluate, genes))
        startup_and_first_batch_seconds = time.perf_counter() - pool_start
        warm_start = time.perf_counter()
        warm_values = np.asarray(pool.map(_process_evaluate, genes))
        warm_batch_seconds = time.perf_counter() - warm_start

    np.testing.assert_array_equal(first_values, serial_values,
                                  err_msg='spawn first batch changed fitness/order')
    np.testing.assert_array_equal(warm_values, serial_values,
                                  err_msg='spawn warm batch changed fitness/order')
    return {
        'status': 'completed',
        'serial_candidate_seconds': serial_seconds,
        'processes': args.processes,
        'serialization_seconds': serialization_seconds,
        'serialized_problem_bytes': len(payload),
        'spawn_startup_and_first_batch_seconds': startup_and_first_batch_seconds,
        'spawn_warm_batch_seconds': warm_batch_seconds,
        'candidate_count': len(genes),
        'payload_array_bytes_not_rss': memory,
        'memory_note': ('Array payload bytes are not RSS or a complete worker-memory bound: '
                        'they exclude Python objects, pickle buffers, temporary arrays, '
                        'allocator overhead, and per-process runtime state.'),
    }


def _independent_fit_worker(payload):
    args_dict, X, y, seed = pickle.loads(payload)
    args = argparse.Namespace(**args_dict)
    local_args = argparse.Namespace(**vars(args))
    local_args.seed = seed
    _, result = _fit_once(local_args, X, y, runner=1, cache_enabled=True)
    return result


def independent_fit_diagnostic(args, X, y):
    """Optional caller-level throughput test: exactly two serial-worker fits."""
    seeds = (args.seed, args.seed + 1)
    serial_start = time.perf_counter()
    serial = []
    for seed in seeds:
        local_args = argparse.Namespace(**vars(args))
        local_args.seed = seed
        _, result = _fit_once(local_args, X, y, runner=1, cache_enabled=True)
        serial.append(result)
    serial_seconds = time.perf_counter() - serial_start

    payloads = [pickle.dumps((vars(args), X, y, seed), protocol=pickle.HIGHEST_PROTOCOL)
                for seed in seeds]
    process_start = time.perf_counter()
    with mp.get_context('spawn').Pool(2) as pool:
        parallel = pool.map(_independent_fit_worker, payloads)
    process_seconds = time.perf_counter() - process_start
    for ix, (expected, actual) in enumerate(zip(serial, parallel)):
        _assert_fit_equal(expected, actual, f'independent process fit {ix}')
    return {
        'two_serial_fits_seconds': serial_seconds,
        'two_process_fits_seconds_including_spawn': process_seconds,
        'throughput_ratio_serial_over_process': serial_seconds / process_seconds if process_seconds else None,
        'note': 'Each fit uses runner=1; this intentionally avoids nested worker pools.',
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--features', type=int, default=8)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--rules', type=int, default=12)
    parser.add_argument('--antecedents', type=int, default=3)
    parser.add_argument('--population', type=int, default=20)
    parser.add_argument('--generations', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=40)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--independent-fits', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    X, y = _make_data(args)
    report = {
        'source': evf.__file__,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pymoo': pymoo.__version__,
        'machine': platform.machine(),
        'settings': vars(args),
        'fit_routes': fit_route_diagnostic(args, X, y),
        'spawn_candidates': process_candidate_diagnostic(args, X, y),
    }
    if args.independent_fits:
        try:
            report['independent_fits'] = independent_fit_diagnostic(args, X, y)
        except Exception as error:
            report['independent_fits'] = {
                'status': 'blocked',
                'error_type': type(error).__name__,
                'error': str(error),
                'blocker_note': ('The independent-fit route could not complete under spawn. '
                                 'The serial and other completed diagnostic results remain valid.'),
            }
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
