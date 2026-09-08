"""Reproducible reference-versus-optimized classification training benchmark.

Run from the repository root after installing ex-fuzzy:
    python benchmarks/benchmark_classifier_fitness.py --samples 1000

Reports median times, checks exact fitness equality and compares seeded fitted
models. No timing assertions: throughput depends on hardware and the workload.
"""
import argparse
import json
import platform
import time
from unittest.mock import patch

import numpy as np
import pymoo
from sklearn.datasets import make_classification

from ex_fuzzy import evolutionary_fit as evf, utils


def run(args):
    X, y = make_classification(
        n_samples=args.samples, n_features=args.features,
        n_informative=args.features - 1, n_redundant=0,
        n_classes=3, random_state=42,
    )
    report = {'python': platform.python_version(), 'numpy': np.__version__,
              'pymoo': pymoo.__version__, 'machine': platform.machine(),
              'settings': vars(args), 'results': []}
    for fixed in (False, True):
        partitions = utils.construct_partitions(X) if fixed else None
        problem = evf.FitRuleBase(X, y, args.rules, 4, 3, linguistic_variables=partitions)
        rng = np.random.default_rng(7)
        genes = rng.integers(problem.xl.astype(int), problem.xu.astype(int) + 1,
                             size=(args.population, problem.n_var))
        evaluation_times, training_times, models, fitness = {}, {}, {}, {}
        for name in ('reference', 'optimized'):
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
            fitness[name] = values
            times = []
            evaluator = (evf.FitRuleBase._evaluate_slow if name == 'reference'
                         else evf.FitRuleBase._evaluate)
            with patch.object(evf.FitRuleBase, '_evaluate', evaluator):
                for _ in range(args.repeats):
                    model = evf.BaseFuzzyRulesClassifier(
                        nRules=args.rules, nAnts=4, linguistic_variables=partitions)
                    start = time.perf_counter()
                    model.fit(X, y, n_gen=args.generations, pop_size=args.population,
                              random_state=7, patience=None)
                    times.append(time.perf_counter() - start)
                models[name] = model
            training_times[name] = float(np.median(times))
        np.testing.assert_array_equal(fitness['reference'], fitness['optimized'])
        left, right = models['reference'], models['optimized']
        assert left.performance == right.performance
        np.testing.assert_array_equal(left.optimization_result_['X'], right.optimization_result_['X'])
        assert left.rule_base.get_consequents() == right.rule_base.get_consequents()
        np.testing.assert_array_equal(left.predict(X), right.predict(X))
        np.testing.assert_array_equal(left.rule_base.get_scores(), right.rule_base.get_scores())
        for a, b in zip(left.rule_base.get_rulebase_matrix(), right.rule_base.get_rulebase_matrix()):
            np.testing.assert_array_equal(a, b)
        report['results'].append({
            'partitions': 'fixed' if fixed else 'optimized',
            'evaluation_seconds': evaluation_times, 'fit_seconds': training_times,
            'evaluation_speedup': evaluation_times['reference'] / evaluation_times['optimized'],
            'fit_speedup': training_times['reference'] / training_times['optimized'],
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
    args = parser.parse_args()
    if args.features < 4 or min(args.samples, args.rules, args.population,
                               args.generations, args.repeats) < 1:
        parser.error('Use at least four features and positive sizes/counts.')
    run(args)
