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
