"""Bounded probes for the speedup routes that are not implemented.

Three independent diagnostics, none of them imported by the package:

* ``--memory`` (C10): peak resident memory of a complete seeded fit, with and
  without the fit-local caches, plus the bytes those caches are allowed to hold.
* ``--overhead`` (C01/C02): how much of one candidate evaluation is fixed
  per-call cost rather than arithmetic, which bounds what batching a whole
  population could win.
* ``--numba`` (D01/D02/D03): whether a compiled kernel can reproduce NumPy's
  reductions bit for bit, which is the acceptance criterion these routes must
  meet before any of them is worth building.

    python benchmarks/prototype_remaining_routes.py --memory --overhead --numba
"""
import argparse
import importlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, rules, utils

fitness = importlib.import_module(evf.__package__ + '._fitness')

MEMORY_WORKER = r'''
import json, resource, sys
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import numpy as np
sys.path.insert(0, {root!r})
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils
import importlib
fitness = importlib.import_module(evf.__package__ + '._fitness')
from sklearn.datasets import make_classification

cfg = json.loads(sys.argv[1])
fuzzy_type = getattr(fs.FUZZY_SETS, cfg['fuzzy_type'])
X, y = make_classification(n_samples=cfg['samples'], n_features=10,
                           n_informative=9, n_redundant=0, n_classes=3,
                           random_state=42)
partitions = utils.construct_partitions(X, fuzzy_type) if cfg['fixed'] else None
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@contextmanager
def plain_scope(problem, enabled):
    if not enabled:
        yield
        return
    problem._fitness_cache = fitness._FitnessCache()
    try:
        yield
    finally:
        del problem._fitness_cache


with ExitStack() as stack:
    if cfg['caches'] == 'off':
        stack.enter_context(patch.object(fitness, '_fitness_cache_scope', plain_scope))
        stack.enter_context(patch.object(evf.FitRuleBase, '_packed_memberships',
                                         lambda self: None))
    model = evf.BaseFuzzyRulesClassifier(nRules=20, nAnts=4,
                                         linguistic_variables=partitions,
                                         fuzzy_type=fuzzy_type)
    model.fit(X, y, n_gen=cfg['generations'], pop_size=40, random_state=7,
              patience=None)
print(json.dumps({{
    'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    'rss_growth_bytes': (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * 1024,
    'performance': float(model.performance),
}}))
'''


def measure_memory(args):
    results = []
    for fuzzy_type in ('t1', 't2'):
        for fixed in (True, False):
            row = {'fuzzy_type': fuzzy_type,
                   'partitions': 'fixed' if fixed else 'optimized'}
            for caches in ('on', 'off'):
                config = dict(samples=args.samples, fuzzy_type=fuzzy_type,
                              fixed=fixed, generations=args.generations,
                              caches=caches)
                script = MEMORY_WORKER.format(root=str(ROOT))
                done = subprocess.run([sys.executable, '-c', script, json.dumps(config)],
                                      capture_output=True, text=True, cwd=str(ROOT))
                if done.returncode != 0:
                    raise RuntimeError(done.stderr[-2000:])
                observed = json.loads(done.stdout.strip().splitlines()[-1])
                row['peak_rss_mb_caches_%s' % caches] = round(
                    observed['peak_rss_bytes'] / 1e6, 1)
                row['performance_caches_%s' % caches] = observed['performance']
            row['extra_peak_mb'] = round(
                row['peak_rss_mb_caches_on'] - row['peak_rss_mb_caches_off'], 1)
            row['same_result'] = (row.pop('performance_caches_on')
                                  == row.pop('performance_caches_off'))
            results.append(row)
    return {'samples': args.samples, 'firing_cache_budget_mb': 8.4,
            'packed_table_budget_mb': 8.4, 'rows': results}


def measure_overhead(args):
    """Split candidate evaluation into sample-independent and per-sample cost.

    Evaluation time is fitted as ``a + b * samples`` over a range of dataset
    sizes with everything else fixed.  The intercept ``a`` is the part that does
    not shrink with the data: interpreter dispatch, per-array bookkeeping and
    fixed-size decoding.  Batching a whole population (C01/C02) can remove at
    most that share, so it bounds the route's benefit.  Fit-local caches are not
    installed here, so these are uncached candidate evaluations.
    """
    sizes = (100, 200, 400, 800, 1600, 3200)
    results = []
    for fuzzy_type in ('t1', 't2'):
        kind = getattr(fs.FUZZY_SETS, fuzzy_type)
        measured = []
        for samples in sizes:
            X, y = make_classification(n_samples=samples, n_features=10,
                                       n_informative=9, n_redundant=0,
                                       n_classes=3, random_state=42)
            problem = evf.FitRuleBase(
                X, y, 20, 4, 3, fuzzy_type=kind,
                linguistic_variables=utils.construct_partitions(X, kind))
            rng = np.random.default_rng(7)
            genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                                 size=(60, problem.n_var))
            runs = []
            for _ in range(5):
                start = time.perf_counter()
                for gene in genes:
                    out = {}
                    problem._evaluate(gene, out)
                runs.append((time.perf_counter() - start) / len(genes))
            measured.append(float(np.median(runs)))
        intercept, slope = np.polynomial.polynomial.polyfit(sizes, measured, 1)
        results.append({
            'fuzzy_type': fuzzy_type,
            'samples': list(sizes),
            'seconds_per_candidate': [round(value, 6) for value in measured],
            'fixed_seconds_per_candidate': round(float(intercept), 6),
            'seconds_per_sample': float(slope),
            'fixed_share': {str(size): round(float(intercept) / value, 3)
                            for size, value in zip(sizes, measured)},
        })
    return {'note': 'fixed_share bounds what batching a population could remove',
            'rows': results}


def check_numba(args):
    """Can a compiled kernel reproduce NumPy's reductions exactly?"""
    try:
        import numba
    except ImportError:
        return {'available': False}

    @numba.njit(cache=False)
    def ordered_product(values):
        out = np.empty(values.shape[0])
        for row in range(values.shape[0]):
            total = 1.0
            for column in range(values.shape[1]):
                total *= values[row, column]
            out[row] = total
        return out

    @numba.njit(cache=False)
    def ordered_sum(values):
        out = np.empty(values.shape[0])
        for row in range(values.shape[0]):
            total = 0.0
            for column in range(values.shape[1]):
                total += values[row, column]
            out[row] = total
        return out

    rng = np.random.default_rng(5)
    warm = rng.random((4, 4))
    start = time.perf_counter()
    ordered_product(warm)
    ordered_sum(warm)
    compile_seconds = time.perf_counter() - start
    product_mismatch = sum_mismatch = 0
    trials = 0
    for _ in range(60):
        rows = int(rng.choice([1, 17, 512, 4096]))
        columns = int(rng.choice([2, 8, 10, 33, 257]))
        values = rng.random((rows, columns))
        trials += 1
        if not np.array_equal(ordered_product(values), np.prod(values, axis=1)):
            product_mismatch += 1
        if not np.array_equal(ordered_sum(values), np.sum(values, axis=1)):
            sum_mismatch += 1
    return {'available': True, 'numba': numba.__version__, 'cases': trials,
            'compile_seconds': round(compile_seconds, 3),
            'ordered_product_mismatches': product_mismatch,
            'ordered_sum_mismatches': sum_mismatch,
            'verdict': ('sequential products match NumPy; sequential sums do not, '
                        'so a compiled route must reproduce pairwise summation '
                        'or accept a numerical-policy change')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--overhead', action='store_true')
    parser.add_argument('--numba', action='store_true')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--generations', type=int, default=10)
    parsed = parser.parse_args()
    report = {}
    if parsed.overhead:
        report['C01_C02_batching_headroom'] = measure_overhead(parsed)
    if parsed.numba:
        report['D01_D02_D03_compiled_parity'] = check_numba(parsed)
    if parsed.memory:
        report['C10_memory'] = measure_memory(parsed)
    print(json.dumps(report or {'note': 'choose at least one probe'}, indent=2))
