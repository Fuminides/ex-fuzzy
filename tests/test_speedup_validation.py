"""End-to-end parity against the full reference, across seeds and score modes."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from benchmark_speedup import compare_outcomes, config
from benchmark_evaluator_variants import _run


@pytest.mark.parametrize('kind,fixed', [('t1', False), ('t1', True), ('t2', False), ('t2', True)])
@pytest.mark.parametrize('seed,mode', [(7, 0), (19, 1), (41, 2)])
def test_complete_search_matches_full_reference(kind, fixed, seed, mode):
    cfg = config(100, kind, fixed, seed, rules=8, antecedents=3,
                 population=12, generations=4, ds_mode=mode,
                 tolerance=0.01, allow_unknown=True)
    reference = _run(dict(cfg, reference=True))
    current = _run(cfg)
    compare_outcomes(reference, current)
    assert current['n_eval'] == 48
    assert len(current['trace']) == 4
    assert len(current['heldout_predictions']) == 101


@pytest.mark.parametrize('key', ['trace', 'population_f', 'chromosome', 'rule_matrices',
                                 'scores', 'heldout_predictions', 'n_eval'])
def test_benchmark_rejects_changed_results(key):
    # Even a one-ULP difference must fail; identical final accuracy is insufficient.
    left = dict(trace=['abc'], population_f=[0.5], chromosome=[1], rule_matrices=[[1]],
                scores=[0.5], heldout_predictions=[1], n_eval=40, seconds=1)
    right = copy.deepcopy(left)
    right[key] = [0.5000000000000001]
    with pytest.raises(AssertionError, match=key):
        compare_outcomes(left, right)


def test_benchmark_requires_trace_and_ignores_only_time():
    compare_outcomes(dict(trace=['abc'], seconds=1), dict(trace=['abc'], seconds=9))
    with pytest.raises(AssertionError, match='Missing'):
        compare_outcomes(dict(trace=[]), dict(trace=[]))


@pytest.mark.parametrize('features', [50, 200])
@pytest.mark.parametrize('fixed', [False, True])
def test_wide_t1_search_matches_full_reference(features, fixed):
    cfg = config(100, 't1', fixed, 19, features=features, rules=8,
                 antecedents=4, population=12, generations=4)
    compare_outcomes(_run(dict(cfg, reference=True)), _run(cfg))


def test_scaling_grid_includes_largest_crossed_workload():
    from types import SimpleNamespace
    from benchmark_speedup import workloads
    grid = workloads(SimpleNamespace(samples=[1000, 10000, 100000],
                                     features=[10, 50, 200], fuzzy_types=['t1']))
    assert len(grid) == 18
    assert {(c['samples'], c['features'], c['fixed']) for c in grid} == {
        (n, d, f) for n in (1000, 10000, 100000) for d in (10, 50, 200)
        for f in (False, True)}
    assert all(c['population'] == 40 and c['generations'] == 5 for c in grid)


def test_resume_preserves_fit_and_invalidates_changed_inputs(tmp_path, monkeypatch):
    import benchmark_speedup as benchmark
    calls = []

    def measured(cfg):
        calls.append(cfg.copy())
        return dict(trace=['exact'], seconds=2.0, n_eval=200)

    monkeypatch.setattr(benchmark, '_run', measured)
    cfg = config(1000, 't1', True, 7)
    saved = benchmark.cached_run(cfg, tmp_path, {'source': 'a'})
    assert benchmark.cached_run(cfg, tmp_path, {'source': 'a'}, resume=True) == saved
    assert len(calls) == 1
    benchmark.cached_run(dict(cfg, features=200), tmp_path, {'source': 'a'}, resume=True)
    benchmark.cached_run(cfg, tmp_path, {'source': 'b'}, resume=True)
    assert len(calls) == 3
    benchmark.cached_run(cfg, tmp_path, {'source': 'a'}, resume=False)
    assert len(calls) == 4


def test_failed_fit_is_not_saved_for_resume(tmp_path, monkeypatch):
    import benchmark_speedup as benchmark

    def failed(cfg):
        raise RuntimeError('worker failed')

    monkeypatch.setattr(benchmark, '_run', failed)
    with pytest.raises(RuntimeError, match='worker failed'):
        benchmark.cached_run(config(1000, 't1', True, 7), tmp_path, {}, resume=True)
    assert not list(tmp_path.glob('*.gz'))
