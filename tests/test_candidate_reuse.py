"""The reuse diagnostic preserves LRU ordering and the observed fit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / 'benchmarks' / 'benchmark_candidate_reuse.py'
spec = importlib.util.spec_from_file_location('candidate_reuse_benchmark', path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_reuse_window_tracks_hits_eviction_and_payload():
    window = benchmark.ReuseWindow(2)
    assert not window.observe(b'a')
    assert not window.observe(b'bb')
    assert window.observe(b'a')
    assert not window.observe(b'ccc')
    assert not window.observe(b'bb')
    result = window.report()
    assert result['hits'] == 1
    assert result['lookups'] == 5
    assert result['peak_key_payload_bytes'] == 5
    assert result['retained_entries'] == 2


def test_observation_preserves_seeded_fit():
    args = SimpleNamespace(samples=60, features=4, rules=5, population=8,
                           generations=2, seed=7, capacity=16)
    result = benchmark.measure(args, True, benchmark.fs.FUZZY_SETS.t1)
    assert result['observed_fit_matches_unobserved']
    assert result['genotype']['lookups'] == 16
    assert result['decoded_feature_partition']['retained_entries'] == 4
