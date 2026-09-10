"""Tests for the population-evaluation route choice.

The scalar and batched evaluators are two implementations of one objective, so
every test here asserts the same thing from a different angle: whichever route
runs, and however it was chosen, the fit is identical.  Speed is measured in
``benchmarks/calibrate_population_dispatch.py``, not asserted here.
"""
import json
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
from pymoo.core.problem import Problem
from sklearn.datasets import load_iris

import evolutionary_fit as evf
import fuzzy_sets as fs
import utils
import _dispatch_profile
from _population_fitness import _RouteProbe, chunk_size, supports_shape


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def test_chunk_size_tracks_the_gather_budget():
    # Eight bytes per sample, rule and feature.  A candidate of exactly the
    # budget admits one; half the budget admits two.
    assert chunk_size(1, 1, 1, gather_budget=8) == 1
    assert chunk_size(1, 1, 1, gather_budget=16) == 2
    assert chunk_size(100, 10, 10, gather_budget=8 * 100 * 10 * 10) == 1


def test_chunk_size_zero_when_one_candidate_exceeds_the_budget():
    assert chunk_size(10_000_000, 100, 100) == 0
    # Degenerate shapes have nothing to gather and cannot batch.
    assert chunk_size(0, 20, 10) == 0
    assert chunk_size(100, 0, 10) == 0


def test_supports_shape_admits_exactly_one_chunk():
    # The historical 512-sample boundary: forty 20-rule, 10-feature candidates
    # fit the budget in a single call and forty-one do not.  Forty-one are still
    # scorable, but only across two chunks.
    assert supports_shape(40, 512, 20, 10)
    assert not supports_shape(41, 512, 20, 10)
    assert chunk_size(512, 20, 10) == 40


def test_supports_shape_rejects_a_lone_candidate():
    # Batching one candidate has no population to amortize over.
    assert not supports_shape(1, 100, 20, 10)


# --------------------------------------------------------------------------
# The probe's decision logic
# --------------------------------------------------------------------------

def _drive(probe, costs, candidates=10):
    """Feed the probe generations whose cost depends only on the route."""
    while True:
        route = probe.route(candidates)
        if route is None:
            return probe.decision
        probe.record(route, costs[route] * candidates, candidates)


def test_probe_prefers_the_cheaper_route():
    assert _drive(_RouteProbe(), {_RouteProbe.SCALAR: 1e-3,
                                  _RouteProbe.BATCH: 1e-4}) == _RouteProbe.BATCH
    assert _drive(_RouteProbe(), {_RouteProbe.SCALAR: 1e-4,
                                  _RouteProbe.BATCH: 1e-3}) == _RouteProbe.SCALAR


def test_probe_does_not_record_its_warmup_generation():
    probe = _RouteProbe()
    first = probe.route(10)
    assert first == _RouteProbe.WARMUP[0]
    probe.record(first, 99.0, 10)
    # The warm-up generation is unlike the ones that follow, so its cost must
    # not reach the comparison at all.
    assert probe.costs() is None
    assert probe.measured()['scalar_candidates'] == 0


def test_probe_settles_on_a_decisive_pair_without_exhausting_the_budget():
    probe = _RouteProbe()
    _drive(probe, {_RouteProbe.SCALAR: 1e-3, _RouteProbe.BATCH: 1e-4})
    # One warm-up plus one counterbalanced pair, not the whole ORDER.
    assert probe.measured()['probing_generations'] == len(_RouteProbe.WARMUP) + 2


def test_probe_keeps_measuring_when_the_routes_are_close():
    probe = _RouteProbe()
    _drive(probe, {_RouteProbe.SCALAR: 1.0e-3, _RouteProbe.BATCH: 0.95e-3})
    assert probe.measured()['probing_generations'] == (
        len(_RouteProbe.WARMUP) + len(_RouteProbe.ORDER))


def test_probe_order_is_counterbalanced():
    # A symmetric order gives both routes the same mean position, so a cache
    # that keeps warming up cannot favour either one.
    order = _RouteProbe.ORDER
    scalar = [i for i, r in enumerate(order) if r == _RouteProbe.SCALAR]
    batch = [i for i, r in enumerate(order) if r == _RouteProbe.BATCH]
    assert len(scalar) == len(batch)
    assert sum(scalar) == sum(batch)


def test_probe_declines_on_populations_too_small_to_measure():
    probe = _RouteProbe()
    assert probe.route(1) is None
    assert probe.decision is None


def test_probe_defaults_to_batching_when_a_route_is_never_measured():
    # Populations can stop being eligible mid-probe; an unmeasured route must
    # not leave the fit without a decision.
    probe = _RouteProbe()
    for _ in range(len(_RouteProbe.WARMUP) + len(_RouteProbe.ORDER)):
        probe.record(_RouteProbe.SCALAR, 1e-3, 10)
    assert probe.decision == _RouteProbe.BATCH


# --------------------------------------------------------------------------
# The stored profile
# --------------------------------------------------------------------------

@pytest.fixture
def profile_path(tmp_path, monkeypatch):
    path = tmp_path / 'dispatch_profile.json'
    monkeypatch.setenv(_dispatch_profile.PROFILE_ENV, str(path))
    return str(path)


def _entry(**overrides):
    entry = {'samples': 400, 'rules': 20, 'features': 10, 'population': 40,
             'decision': 'batch'}
    entry.update(overrides)
    return entry


def test_profile_round_trips(profile_path):
    _dispatch_profile.save([_entry()])
    assert _dispatch_profile.decision_for(400, 20, 10, 40) == 'batch'


def test_profile_default_path_follows_the_environment(profile_path):
    assert _dispatch_profile.default_path() == profile_path


def test_missing_profile_yields_no_decision(profile_path):
    assert _dispatch_profile.decision_for(400, 20, 10, 40) is None


def test_unreadable_profile_yields_no_decision(profile_path):
    with open(profile_path, 'w') as handle:
        handle.write('this is not json')
    assert _dispatch_profile.decision_for(400, 20, 10, 40) is None


def test_profile_from_other_hardware_is_ignored(profile_path):
    _dispatch_profile.save([_entry()])
    with open(profile_path) as handle:
        payload = json.load(handle)
    payload['fingerprint']['machine'] = 'a-different-machine'
    with open(profile_path, 'w') as handle:
        json.dump(payload, handle)
    assert _dispatch_profile.decision_for(400, 20, 10, 40) is None


def test_profile_of_another_format_version_is_ignored(profile_path):
    _dispatch_profile.save([_entry()])
    with open(profile_path) as handle:
        payload = json.load(handle)
    payload['version'] = _dispatch_profile.FORMAT_VERSION + 1
    with open(profile_path, 'w') as handle:
        json.dump(payload, handle)
    assert _dispatch_profile.decision_for(400, 20, 10, 40) is None


def test_profile_lookup_uses_the_nearest_calibrated_workload():
    entries = [_entry(samples=200, decision='batch'),
               _entry(samples=800, decision='scalar')]
    assert _dispatch_profile.lookup(entries, 220, 20, 10, 40) == 'batch'
    assert _dispatch_profile.lookup(entries, 780, 20, 10, 40) == 'scalar'


def test_profile_lookup_declines_a_distant_workload():
    entries = [_entry(samples=200)]
    tolerance = _dispatch_profile.NEIGHBOUR_TOLERANCE
    assert _dispatch_profile.lookup(entries, int(200 * tolerance), 20, 10, 40)
    assert _dispatch_profile.lookup(
        entries, int(200 * tolerance * 2), 20, 10, 40) is None
    # Distance in any single dimension is enough to decline.
    assert _dispatch_profile.lookup(entries, 200, 20, 10, 4000) is None


def test_profile_lookup_ignores_malformed_entries():
    entries = [{'samples': 'not a number'}, _entry(decision='scalar')]
    assert _dispatch_profile.lookup(entries, 400, 20, 10, 40) == 'scalar'


def test_profile_lookup_rejects_an_unknown_decision():
    assert _dispatch_profile.lookup(
        [_entry(decision='something else')], 400, 20, 10, 40) is None


# --------------------------------------------------------------------------
# End to end: the route must never change a fit
# --------------------------------------------------------------------------

def _fitted(route, samples=None, profile=None):
    X, y = load_iris(return_X_y=True)
    if samples is not None:
        # Tile the data to reach a size where the routes trade places.
        repeats = int(np.ceil(samples / len(X)))
        X = np.tile(X, (repeats, 1))[:samples]
        y = np.tile(y, repeats)[:samples]
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)

    if route == 'scalar':
        context = patch.object(evf.FitRuleBase, '_evaluate_elementwise',
                               Problem._evaluate_elementwise)
    elif route == 'batch':
        def always_batch(self, population):
            probe = _RouteProbe()
            probe.decision = _RouteProbe.BATCH
            return probe
        context = patch.object(evf.FitRuleBase, '_route_probe_state',
                               always_batch)
    else:
        context = nullcontext()

    with context:
        model = evf.BaseFuzzyRulesClassifier(
            nRules=15, nAnts=3, linguistic_variables=partitions,
            fuzzy_type=fs.FUZZY_SETS.t1)
        model.fit(X, y, n_gen=8, pop_size=20, random_state=5)
    return (float(model.performance),
            np.asarray(model.optimization_result_['X']).tolist(),
            np.asarray(model.rule_base.get_scores()).tolist(),
            np.asarray(model.rule_base.get_rulebase_matrix()[0]).tolist(),
            model.predict(X).tolist(),
            int(model.optimization_result_['algorithm'].evaluator.n_eval))


@pytest.mark.parametrize('samples', [None, 700])
def test_every_route_produces_the_identical_fit(samples):
    # 700 samples needs more than one chunk, so this also covers chunking.
    reference = _fitted('scalar', samples)
    assert _fitted('batch', samples) == reference
    assert _fitted(None, samples) == reference


def test_a_stored_profile_does_not_change_the_fit(profile_path):
    reference = _fitted(None)
    for decision in ('batch', 'scalar'):
        _dispatch_profile.save([_entry(samples=150, rules=15, features=4,
                                       population=20, decision=decision)])
        assert _fitted(None) == reference


def test_a_stored_profile_settles_the_probe_before_it_measures(profile_path):
    _dispatch_profile.save([_entry(samples=150, rules=15, features=4,
                                   population=20, decision='scalar')])
    seen = []
    original = evf.FitRuleBase._route_probe_state

    def spy(self, population):
        probe = original(self, population)
        seen.append(probe.measured())
        return probe

    with patch.object(evf.FitRuleBase, '_route_probe_state', spy):
        _fitted(None)
    assert seen, 'the population evaluator never ran'
    assert seen[0]['decision'] == _RouteProbe.SCALAR
    assert seen[0]['probing_generations'] == 0


def test_the_probe_reaches_a_decision_during_an_ordinary_fit():
    seen = []
    original = evf.FitRuleBase._route_probe_state

    def spy(self, population):
        probe = original(self, population)
        seen.append(probe)
        return probe

    with patch.object(evf.FitRuleBase, '_route_probe_state', spy):
        _fitted(None)
    assert seen and seen[-1].decision in (_RouteProbe.SCALAR, _RouteProbe.BATCH)


def test_the_route_probe_does_not_outlive_the_fit():
    # The probe is fit-local, like the caches it sits beside.
    X, y = load_iris(return_X_y=True)
    partitions = utils.construct_partitions(X, fs.FUZZY_SETS.t1)
    model = evf.BaseFuzzyRulesClassifier(
        nRules=10, nAnts=3, linguistic_variables=partitions,
        fuzzy_type=fs.FUZZY_SETS.t1)
    model.fit(X, y, n_gen=3, pop_size=10, random_state=5)
    problem = getattr(model, 'problem_', None)
    if problem is not None:
        assert not hasattr(problem, '_route_probe')
