"""Fresh-process complete-fit comparison of the classification evaluator variants.

Every variant runs one seeded fit in its own process, in randomized order, so
import cost, allocator state and measurement order cannot favour one variant.
The driver checks that all variants produce the identical fitness, chromosome,
rule scores and predictions before reporting any timing.

    python benchmarks/benchmark_evaluator_variants.py --samples 1000 --repeats 3
    python benchmarks/benchmark_evaluator_variants.py --fuzzy-type t2 --variants baseline current

Variants are benchmark-only switches; they do not change public classifier
options.  ``baseline`` disables every evaluator change these variants isolate.
Explicit ``--variants current compiled-cold compiled-warm`` compares the optional
Numba prototype with and without first-use compilation inside the fit timer.
Warm compilation/parity checks happen before timing; neither variant is enabled
by default or imported by production code.
"""
import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: Each variant lists the optimizations it turns OFF.
VARIANTS = {
    'baseline': ('arrays', 'dominance', 'firing-cache'),
    'dominance-only': ('arrays', 'firing-cache'),
    'arrays-only': ('dominance', 'firing-cache'),
    'no-firing-cache': ('firing-cache',),
    'current': (),
}
DEFAULT_VARIANTS = list(VARIANTS)
# Optional benchmark-only dispatch; ordinary runs do not require Numba.
VARIANTS.update({'compiled-cold': (), 'compiled-warm': ()})

WORKER = r'''
import json, sys, time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
sys.path.insert(0, {root!r})
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils
import importlib
fitness = importlib.import_module(evf.__package__ + '._fitness')
from sklearn.datasets import make_classification

def _no_firing_cache_scope(fitness):
    from contextlib import contextmanager

    @contextmanager
    def scope(problem, enabled):
        if not enabled:
            yield
            return
        problem._fitness_cache = fitness._FitnessCache()
        try:
            yield
        finally:
            del problem._fitness_cache
    return scope


cfg = json.loads(sys.argv[1])
fuzzy_type = getattr(fs.FUZZY_SETS, cfg['fuzzy_type'])
X, y = make_classification(
    n_samples=cfg['samples'], n_features=cfg['features'],
    n_informative=cfg['features'] - 1, n_redundant=0, n_classes=3,
    random_state=42)
partitions = utils.construct_partitions(X, fuzzy_type) if cfg['fixed'] else None

with ExitStack() as stack:
    if 'arrays' in cfg['disable']:
        stack.enter_context(patch.object(evf.FitRuleBase, 'array_evaluation', False))
    if 'dominance' in cfg['disable']:
        stack.enter_context(patch.object(fitness, '_dominance',
                                         fitness._dominance_reference))
    if 'firing-cache' in cfg['disable']:
        stack.enter_context(patch.object(fitness, '_FiringCache',
                                         lambda *a, **k: (_ for _ in ()).throw(TypeError)))
        stack.enter_context(patch.object(
            fitness, '_fitness_cache_scope', _no_firing_cache_scope(fitness)))
    if cfg.get('compiled'):
        sys.path.insert(0, {benchmarks!r})
        import prototype_exact_compiled_reductions as compiled
        if compiled.KERNELS is None:
            raise RuntimeError('compiled variants require optional Numba')
        if cfg['compiled'] == 'compiled-warm':
            parity = compiled.parity_checks()
            if any(parity[key] for key in (
                    'firing_mismatches', 'dominance_mismatches', 'fallback_mismatches')):
                raise RuntimeError('compiled kernel parity failed')
        arrfit = importlib.import_module(evf.__package__ + '._array_fitness')
        stack.enter_context(patch.object(
            arrfit, 'firing_strengths',
            compiled._compiled_array_firing(arrfit.firing_strengths)))
        stack.enter_context(patch.object(
            arrfit, '_dominance', compiled.compiled_dominance))
    model = evf.BaseFuzzyRulesClassifier(
        nRules=cfg['rules'], nAnts=cfg['antecedents'],
        linguistic_variables=partitions, fuzzy_type=fuzzy_type)
    start = time.perf_counter()
    model.fit(X, y, n_gen=cfg['generations'], pop_size=cfg['population'],
              random_state=7, patience=None)
    elapsed = time.perf_counter() - start

print(json.dumps({{
    'seconds': elapsed,
    'performance': float(model.performance),
    'population_x': np.asarray(model.optimization_result_['algorithm'].pop.get('X')).tolist(),
    'population_f': np.asarray(model.optimization_result_['algorithm'].pop.get('F')).tolist(),
    'n_eval': int(model.optimization_result_['algorithm'].evaluator.n_eval),
    'chromosome': [int(v) for v in np.asarray(model.optimization_result_['X'])],
    'consequents': [int(v) for v in model.rule_base.get_consequents()],
    'scores': [float(v) for v in np.asarray(model.rule_base.get_scores())],
    'predictions': [int(v) for v in np.asarray(model.predict(X))],
}}))
'''


def _run(config):
    script = WORKER.format(root=str(ROOT), benchmarks=str(ROOT / 'benchmarks'))
    result = subprocess.run([sys.executable, '-c', script, json.dumps(config)],
                            capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def run(args):
    report = {'python': platform.python_version(), 'numpy': np.__version__,
              'machine': platform.machine(), 'processor': platform.processor(),
              'settings': vars(args), 'results': []}
    selected = [name for name in args.variants if name in VARIANTS]
    if len(selected) != len(args.variants):
        raise SystemExit('unknown variant; choose from %s' % ', '.join(VARIANTS))
    for fixed in (False, True):
        base = dict(samples=args.samples, features=args.features, rules=args.rules,
                    antecedents=args.antecedents, population=args.population,
                    generations=args.generations, fuzzy_type=args.fuzzy_type,
                    fixed=fixed)
        timings = {name: [] for name in selected}
        outcomes = {}
        schedule = [(name, index) for name in selected for index in range(args.repeats)]
        random.Random(args.seed).shuffle(schedule)
        for name, _ in schedule:
            observed = _run(dict(base, disable=list(VARIANTS[name]),
                                 compiled=name if name.startswith('compiled-') else None))
            timings[name].append(observed.pop('seconds'))
            previous = outcomes.setdefault(name, observed)
            if previous != observed:
                raise SystemExit('variant %r changed the fitted model between repeats' % name)
        expected = outcomes[selected[-1]]
        for name in selected:
            if outcomes[name] != expected:
                raise SystemExit('variant %r changed the fitted model' % name)
        entry = {'partitions': 'fixed' if fixed else 'optimized', 'exact_parity': True,
                 'variants': {}}
        reference = statistics.median(timings[selected[0]])
        for name in selected:
            times = timings[name]
            entry['variants'][name] = {
                'median_seconds': statistics.median(times),
                'min_seconds': min(times), 'max_seconds': max(times),
                'speedup_vs_%s' % selected[0]: reference / statistics.median(times),
            }
        report['results'].append(entry)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--features', type=int, default=10)
    parser.add_argument('--rules', type=int, default=20)
    parser.add_argument('--antecedents', type=int, default=4)
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--generations', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--fuzzy-type', choices=['t1', 't2'], default='t1')
    parser.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS)
    run(parser.parse_args())
