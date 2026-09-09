"""Reproducible reference-versus-optimized classification training benchmark.

Run from the repository root after installing ex-fuzzy:
    python benchmarks/benchmark_classifier_fitness.py --samples 1000

Reports median times, checks exact fitness equality and compares seeded fitted
models. No timing assertions: throughput depends on hardware and the workload.
Standalone candidate timings are uncached. Complete fits compare the full
reference, uncached optimized objective and fit-local memoization separately.
"""
import argparse
from contextlib import ExitStack, nullcontext
import importlib
import json
import platform
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pymoo
from sklearn.datasets import make_classification

# Always benchmark this checkout, even if an older release is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, rules, utils
fitness_module = importlib.import_module(evf.__package__ + '._fitness')


def _legacy_delete_rule_duplicates(self, list_rules):
    """Reference duplicate lookup before the exact A11 micro-optimization."""
    unique = {}
    for ix, rule in enumerate(list_rules):
        try:
            unique[rule]
        except KeyError:
            unique[rule] = ix
    return [list_rules[x] for x in unique.values()]


def run(args):
    fuzzy_type = getattr(fs.FUZZY_SETS, args.fuzzy_type)
    X, y = make_classification(
        n_samples=args.samples, n_features=args.features,
        n_informative=args.features - 1, n_redundant=0,
        n_classes=3, random_state=42,
    )
    report = {'python': platform.python_version(), 'numpy': np.__version__,
              'pymoo': pymoo.__version__, 'machine': platform.machine(),
              'source': evf.__file__, 'settings': vars(args), 'results': []}
    for fixed in (False, True):
        partitions = utils.construct_partitions(X, fuzzy_type) if fixed else None
        problem = evf.FitRuleBase(X, y, args.rules, 4, 3, linguistic_variables=partitions,
                                 fuzzy_type=fuzzy_type)
        rng = np.random.default_rng(7)
        genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                             size=(args.population, problem.n_var))
        evaluation_times, training_times, models, fitness = {}, {}, {}, {}
        evaluation_runs, training_runs = {}, {}
        for name in ('reference', 'uncached', 'optimized'):
            method = problem._evaluate_slow if name == 'reference' else problem._evaluate
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                values = []
                for gene in genes:
                    out = {}
                    method(gene, out)
                    values.append(out['F'])
                times.append(time.perf_counter() - start)
            evaluation_times[name] = float(np.median(times))
            evaluation_runs[name] = times
            fitness[name] = values
            times = []
            evaluator = (evf.FitRuleBase._evaluate_slow if name == 'reference'
                         else evf.FitRuleBase._evaluate)
            cache_scope = (patch.object(fitness_module, '_fitness_cache_scope',
                                        lambda *a: nullcontext())
                           if name == 'uncached' else nullcontext())
            with cache_scope, patch.object(evf.FitRuleBase, '_evaluate', evaluator):
                for _ in range(args.repeats):
                    model = evf.BaseFuzzyRulesClassifier(
                        nRules=args.rules, nAnts=4, linguistic_variables=partitions,
                        fuzzy_type=fuzzy_type)
                    start = time.perf_counter()
                    model.fit(X, y, n_gen=args.generations, pop_size=args.population,
                              random_state=7, patience=None)
                    times.append(time.perf_counter() - start)
                models[name] = model
            training_times[name] = float(np.median(times))
            training_runs[name] = times
        for name in ('reference', 'uncached'):
            np.testing.assert_array_equal(fitness[name], fitness['optimized'])
            left, right = models[name], models['optimized']
            assert left.performance == right.performance
            np.testing.assert_array_equal(left.optimization_result_['X'], right.optimization_result_['X'])
            assert left.rule_base.get_consequents() == right.rule_base.get_consequents()
            np.testing.assert_array_equal(left.predict(X), right.predict(X))
            np.testing.assert_array_equal(left.rule_base.get_scores(), right.rule_base.get_scores())
            assert left.optimization_result_['algorithm'].evaluator.n_eval == right.optimization_result_['algorithm'].evaluator.n_eval
            for a, b in zip(left.rule_base.get_rulebase_matrix(), right.rule_base.get_rulebase_matrix()):
                np.testing.assert_array_equal(a, b)
        report['results'].append({
            'partitions': 'fixed' if fixed else 'optimized',
            'evaluation_seconds': evaluation_times, 'fit_seconds': training_times,
            'evaluation_runs_seconds': evaluation_runs, 'fit_runs_seconds': training_runs,
            'evaluation_speedup': evaluation_times['reference'] / evaluation_times['optimized'],
            'fit_speedup': training_times['reference'] / training_times['optimized'],
            'cache_fit_speedup': training_times['uncached'] / training_times['optimized'],
            'exact_parity': True,
        })
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--features', type=int, default=10)
    parser.add_argument('--rules', type=int, default=20)
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--generations', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--fuzzy-type', choices=['t1', 't2'], default='t1')
    parser.add_argument('--legacy-firing', action='store_true',
                        help='Benchmark the previous per-rule firing implementation.')
    parser.add_argument('--legacy-masks', action='store_true',
                        help='Benchmark without cached consequent-match masks.')
    parser.add_argument('--legacy-duplicate-lookup', action='store_true',
                        help='Benchmark the previous duplicate-rule lookup.')
    parser.add_argument('--legacy-dominance', action='store_true',
                        help='Benchmark the per-rule dominance reductions.')
    parser.add_argument('--legacy-arrays', action='store_true',
                        help='Benchmark the rule-object evaluator instead of the array one.')
    args = parser.parse_args()
    if args.features < 4 or min(args.samples, args.rules, args.population,
                               args.generations, args.repeats) < 1:
        parser.error('Use at least four features and positive sizes/counts.')
    with ExitStack() as stack:
        if args.legacy_firing:
            stack.enter_context(patch.object(rules, '_gather_rule_firing', lambda *a: None))
        if args.legacy_masks:
            stack.enter_context(patch.object(fitness_module, '_ClassMaskCache', lambda y: None))
        if args.legacy_duplicate_lookup:
            stack.enter_context(patch.object(rules.RuleBase, 'delete_rule_duplicates',
                                              _legacy_delete_rule_duplicates))
        if args.legacy_dominance:
            stack.enter_context(patch.object(fitness_module, '_dominance',
                                             fitness_module._dominance_reference))
        if args.legacy_arrays:
            stack.enter_context(patch.object(evf.FitRuleBase, 'array_evaluation', False))
        run(args)
