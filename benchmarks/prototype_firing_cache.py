"""B04/B05 probe: does caching per-rule firing columns pay off?

Not imported by the package.  It monkeypatches the array evaluator's firing
function with a bounded fit-local cache keyed on the effective antecedents,
checks that every candidate keeps its exact objective, and reports hit rates,
retained bytes and candidate-evaluation time against the production path.

    python benchmarks/prototype_firing_cache.py --samples 1000
    python benchmarks/prototype_firing_cache.py --samples 1000 --fuzzy-type t2

Only fixed partitions can reuse firing across candidates: with optimized
partitions (B05) every chromosome carries its own memberships, so the same
antecedent indexes denote different fuzzy sets.  The script reports that case
separately to show the reuse is not transferable.
"""
import argparse
import importlib
import json
from collections import OrderedDict
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, rules, utils

arrfit = importlib.import_module(evf.__package__ + '._array_fitness')


class FiringCache:
    """Bounded LRU of firing columns keyed on one rule's effective antecedents."""

    def __init__(self, max_bytes):
        self.entries = OrderedDict()
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.peak_bytes = 0

    def take(self, key):
        column = self.entries.get(key)
        if column is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return column

    def store(self, key, column):
        self.entries[key] = column
        self.bytes += column.nbytes
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        while self.bytes > self.max_bytes and len(self.entries) > 1:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.nbytes


def cached_firing(cache):
    def firing_strengths(antecedents, truth, n_samples, interval):
        tail = (2,) if interval else ()
        keys = [row.tobytes() for row in antecedents]
        columns = [cache.take(key) for key in keys]
        missing = [index for index, column in enumerate(columns) if column is None]
        if missing:
            computed = rules._gather_firing_from_arrays(
                antecedents[missing], truth, n_samples, tail)
            if computed is None:
                return None
            for slot, index in enumerate(missing):
                column = np.ascontiguousarray(computed[:, slot])
                columns[index] = column
                cache.store(keys[index], column)
        result = np.empty((n_samples, len(columns)) + tail)
        for index, column in enumerate(columns):
            result[:, index] = column
        return result
    return firing_strengths


def run_fits(args):
    """Measure complete seeded fits, where GA offspring share parent rules."""
    fuzzy_type = getattr(fs.FUZZY_SETS, args.fuzzy_type)
    X, y = make_classification(
        n_samples=args.samples, n_features=args.features,
        n_informative=args.features - 1, n_redundant=0, n_classes=3,
        random_state=42)
    report = {'settings': vars(args), 'mode': 'complete fits', 'results': []}
    for fixed in (True, False):
        partitions = utils.construct_partitions(X, fuzzy_type) if fixed else None
        timings, outcomes, caches = {}, {}, {}
        for label in ('production', 'cached'):
            cache = FiringCache(args.max_bytes)
            runs = []
            for _ in range(args.repeats):
                cache.entries.clear()
                cache.bytes = cache.hits = cache.misses = 0
                context = (patch.object(arrfit, 'firing_strengths', cached_firing(cache))
                           if label == 'cached' else
                           patch.object(arrfit, 'firing_strengths', arrfit.firing_strengths))
                model = evf.BaseFuzzyRulesClassifier(
                    nRules=args.rules, nAnts=4, linguistic_variables=partitions,
                    fuzzy_type=fuzzy_type)
                start = time.perf_counter()
                with context:
                    model.fit(X, y, n_gen=args.generations, pop_size=args.population,
                              random_state=7, patience=None)
                runs.append(time.perf_counter() - start)
            timings[label] = float(np.median(runs))
            caches[label] = cache
            outcomes[label] = (float(model.performance),
                               [int(v) for v in np.asarray(model.optimization_result_['X'])],
                               [int(v) for v in np.asarray(model.predict(X))])
        cache = caches['cached']
        lookups = cache.hits + cache.misses
        report['results'].append({
            'partitions': 'fixed' if fixed else 'optimized',
            'exact_parity': outcomes['production'] == outcomes['cached'],
            'production_seconds': round(timings['production'], 4),
            'cached_seconds': round(timings['cached'], 4),
            'speedup': round(timings['production'] / timings['cached'], 3),
            'hit_rate': round(cache.hits / lookups, 4) if lookups else 0.0,
            'peak_cached_bytes': cache.peak_bytes,
        })
    print(json.dumps(report, indent=2))


def run(args):
    fuzzy_type = getattr(fs.FUZZY_SETS, args.fuzzy_type)
    X, y = make_classification(
        n_samples=args.samples, n_features=args.features,
        n_informative=args.features - 1, n_redundant=0, n_classes=3,
        random_state=42)
    report = {'settings': vars(args), 'results': []}
    for fixed in (True, False):
        partitions = utils.construct_partitions(X, fuzzy_type) if fixed else None
        problem = evf.FitRuleBase(X, y, args.rules, 4, 3,
                                  linguistic_variables=partitions,
                                  fuzzy_type=fuzzy_type)
        rng = np.random.default_rng(7)
        genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                             size=(args.candidates, problem.n_var))
        cache = FiringCache(args.max_bytes)
        timings, values = {}, {}
        for label in ('production', 'cached'):
            context = (patch.object(arrfit, 'firing_strengths', cached_firing(cache))
                       if label == 'cached' else patch.object(
                           arrfit, 'firing_strengths', arrfit.firing_strengths))
            runs = []
            for _ in range(args.repeats):
                cache.entries.clear()
                cache.bytes = cache.hits = cache.misses = 0
                start = time.perf_counter()
                with context:
                    observed = []
                    for gene in genes:
                        out = {}
                        problem._evaluate(gene, out)
                        observed.append(out['F'])
                runs.append(time.perf_counter() - start)
            timings[label] = float(np.median(runs))
            values[label] = observed
        # Reusing a firing column across candidates is only valid while the
        # memberships are fixed. The optimized-partition row is expected to
        # differ: it is evidence against B05, not a production defect.
        exact = bool(np.array_equal(values['production'], values['cached']))
        differing = int(np.count_nonzero(
            np.asarray(values['production']) != np.asarray(values['cached'])))
        lookups = cache.hits + cache.misses
        report['results'].append({
            'partitions': 'fixed' if fixed else 'optimized',
            'exact_parity': exact,
            'candidates_with_a_different_objective': differing,
            'production_seconds': round(timings['production'], 4),
            'cached_seconds': round(timings['cached'], 4),
            'speedup': round(timings['production'] / timings['cached'], 3),
            'hit_rate': round(cache.hits / lookups, 4) if lookups else 0.0,
            'peak_cached_bytes': cache.peak_bytes,
        })
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--features', type=int, default=10)
    parser.add_argument('--rules', type=int, default=20)
    parser.add_argument('--candidates', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-bytes', type=int, default=8 * 1024 * 1024)
    parser.add_argument('--fuzzy-type', choices=['t1', 't2'], default='t1')
    parser.add_argument('--fits', action='store_true',
                        help='Measure complete seeded fits instead of loose candidates.')
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--generations', type=int, default=20)
    parsed = parser.parse_args()
    (run_fits if parsed.fits else run)(parsed)
