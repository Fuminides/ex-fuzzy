"""Measure B01 reuse in a real seeded search without changing its population.

Run: python benchmarks/benchmark_candidate_reuse.py --generations 30
The extra decoding/key analysis happens after fitting. Cache hits are simulated;
no production cache is enabled and estimated savings exclude cache overhead.
"""
import argparse
from collections import OrderedDict
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import pymoo
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils
fitness_module = importlib.import_module(evf.__package__ + '._fitness')


class ReuseWindow:
    """Simulate an entry-bounded LRU; count retained key payload bytes."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0
        self.lookups = 0
        self.peak_key_bytes = 0
        self.key_bytes = 0

    def observe(self, key):
        self.lookups += 1
        hit = key in self.entries
        if hit:
            self.hits += 1
            self.entries.move_to_end(key)
        else:
            self.entries[key] = None
            self.key_bytes += len(key)
            if len(self.entries) > self.capacity:
                removed, _ = self.entries.popitem(last=False)
                self.key_bytes -= len(removed)
            self.peak_key_bytes = max(self.peak_key_bytes, self.key_bytes)
        return hit

    def report(self):
        return dict(lookups=self.lookups, hits=self.hits,
                    hit_rate=self.hits / self.lookups if self.lookups else 0.,
                    retained_entries=len(self.entries),
                    peak_key_payload_bytes=self.peak_key_bytes)


def partition_keys(base, kind):
    """Exact decoded parameter keys for the benchmark's built-in T1/T2 sets."""
    result = []
    for variable in base.antecedents:
        terms = []
        for term in variable.linguistic_variables:
            parameters = ([term.membership_parameters] if kind == fs.FUZZY_SETS.t1
                          else [term.secondMF_lower, term.secondMF_upper])
            terms.append((tuple(term.domain), parameters))
        result.append(repr(terms).encode())
    return result


def measure(args, fixed, kind):
    X, y = make_classification(n_samples=args.samples, n_features=args.features,
                               n_informative=args.features - 1, n_redundant=0,
                               n_classes=3, random_state=42)
    partitions = utils.construct_partitions(X, kind) if fixed else None
    trace = []
    evaluate = evf.FitRuleBase._evaluate

    def observed(problem, gene, out, *a, **kw):
        snapshot = np.asarray(gene).copy()
        start = time.perf_counter()
        evaluate(problem, gene, out, *a, **kw)
        trace.append((snapshot, time.perf_counter() - start))

    model = evf.BaseFuzzyRulesClassifier(nRules=args.rules, nAnts=4,
                                        linguistic_variables=partitions, fuzzy_type=kind)
    with patch.object(fitness_module, '_fitness_cache_scope', lambda *a: nullcontext()):
        baseline = evf.BaseFuzzyRulesClassifier(nRules=args.rules, nAnts=4,
                                               linguistic_variables=partitions, fuzzy_type=kind)
        baseline.fit(X, y, n_gen=args.generations, pop_size=args.population,
                     random_state=args.seed, patience=None)
        with patch.object(evf.FitRuleBase, '_evaluate', observed):
            model.fit(X, y, n_gen=args.generations, pop_size=args.population,
                      random_state=args.seed, patience=None)
    np.testing.assert_array_equal(baseline.optimization_result_['X'], model.optimization_result_['X'])
    np.testing.assert_array_equal(baseline.predict(X), model.predict(X))
    np.testing.assert_array_equal(baseline.rule_base.get_scores(), model.rule_base.get_scores())
    problem = evf.FitRuleBase(X, y, args.rules, 4, 3, linguistic_variables=partitions,
                              fuzzy_type=kind)
    raw, partition, firing = [ReuseWindow(args.capacity) for _ in range(3)]
    avoided_seconds = raw_key_seconds = decoded_key_seconds = decode_seconds = 0.
    for gene, duration in trace:
        start = time.perf_counter()
        # Include dtype and shape; do not merge merely equivalent genotypes.
        key = repr((gene.dtype.str, gene.shape)).encode() + gene.tobytes()
        if raw.observe(key):
            avoided_seconds += duration
        raw_key_seconds += time.perf_counter() - start
        start = time.perf_counter()
        base = problem._construct_ruleBase(gene, kind)
        decode_seconds += time.perf_counter() - start
        start = time.perf_counter()
        keys = partition_keys(base, kind)
        for index, key in enumerate(keys):
            partition.observe(repr(index).encode() + b':' + key)
        for rule in base.get_rules():
            # Firing depends on active feature partitions, not consequents or
            # weights. The benchmark generates no modifiers/custom t-norms.
            active = [(i, term, keys[i]) for i, term in enumerate(rule.antecedents)
                      if term >= 0]
            firing.observe(repr(active).encode())
        decoded_key_seconds += time.perf_counter() - start
    return dict(partitions='fixed' if fixed else 'optimized', fuzzy_type=kind.name,
                genotype=raw.report(), decoded_feature_partition=partition.report(),
                decoded_rule_firing=firing.report(),
                genotype_key_and_lookup_seconds=raw_key_seconds,
                decoded_key_and_lookup_seconds=decoded_key_seconds,
                extra_decode_seconds=decode_seconds, observed_fit_matches_unobserved=True,
                evaluated_candidate_seconds=sum(t for _, t in trace),
                genotype_avoided_evaluation_seconds_estimate=avoided_seconds,
                trace_array_bytes=sum(g.nbytes for g, _ in trace),
                full_firing_cache_value_bytes_estimate=args.capacity * args.samples * 8 *
                (2 if kind == fs.FUZZY_SETS.t2 else 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('samples', 1000), ('features', 10), ('rules', 20),
                          ('population', 40), ('generations', 30), ('capacity', 256)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--fuzzy-type', choices=['t1', 't2'], default='t1')
    args = parser.parse_args()
    if args.features < 4 or min(args.samples, args.rules, args.population,
                               args.generations, args.capacity) < 1:
        parser.error('Use at least four features and positive sizes/counts.')
    kind = getattr(fs.FUZZY_SETS, args.fuzzy_type)
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                         pymoo=pymoo.__version__, source=evf.__file__, settings=vars(args),
                         results=[measure(args, fixed, kind) for fixed in (False, True)]),
                     indent=2))
