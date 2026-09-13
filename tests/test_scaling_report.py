"""A long campaign must not publish a partial or inconsistent README result."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import publish_scaling_report as publisher


@pytest.fixture
def report():
    rows = []
    for samples in (1000, 10000, 100000):
        for features in (10, 50, 200):
            for fixed in (False, True):
                rows.append(dict(samples=samples, fixed=fixed, fuzzy_type='t1',
                    settings=dict(samples=samples, features=features, fixed=fixed,
                                  fuzzy_type='t1', rules=20, antecedents=4,
                                  population=40, generations=5),
                    exact_parity=True, speedup=2.0,
                    seconds={'Ex-Fuzzy 2.0': [3., 4., 5.], 'Ex-Fuzzy 3.0': [1., 2., 3.]},
                    median_seconds={'Ex-Fuzzy 2.0': 4., 'Ex-Fuzzy 3.0': 2.},
                    evidence={str(seed): dict(n_eval=200, trace=['0' * 64] * 5)
                              for seed in (7, 19, 41)}))
    return dict(results=rows, seeds=[7, 19, 41], source_sha256='fixture')


@pytest.mark.parametrize('damage', ['missing_large_case', 'duplicate_case', 'parity_failure',
                                   'missing_repeat', 'wrong_median', 'wrong_ratio',
                                   'short_trace', 'short_search'])
def test_refuses_incomplete_or_inconsistent_report(report, damage):
    row = report['results'][-1]
    if damage == 'missing_large_case':
        report['results'].pop()
    elif damage == 'duplicate_case':
        report['results'][-1] = copy.deepcopy(report['results'][0])
    elif damage == 'parity_failure':
        row['exact_parity'] = False
    elif damage == 'missing_repeat':
        row['seconds']['Ex-Fuzzy 2.0'].pop()
    elif damage == 'wrong_median':
        row['median_seconds']['Ex-Fuzzy 2.0'] = 3.0
    elif damage == 'wrong_ratio':
        row['speedup'] = 99.0
    elif damage == 'short_trace':
        row['evidence']['7']['trace'].pop()
    elif damage == 'short_search':
        row['evidence']['7']['n_eval'] = 199
    with pytest.raises(ValueError):
        publisher.validate(report)


def test_publish_preserves_existing_readme_and_is_repeatable(report, tmp_path, monkeypatch):
    path = tmp_path / 't1_scaling.json'
    path.write_text(json.dumps(report))
    readme = tmp_path / 'README.md'
    readme.write_bytes(b'Existing text\r\n### Backend Comparison\r\nOther content\r\n')
    monkeypatch.setattr(publisher, 'source_digest', lambda: 'fixture')
    monkeypatch.setattr(publisher, 'plot', lambda report, destination:
                        destination.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>'))
    publisher.publish(path, readme)
    first = readme.read_bytes()
    assert first.startswith(b'Existing text\r\n')
    assert first.endswith(b'### Backend Comparison\r\nOther content\r\n')
    assert b'108 complete fits' in first
    assert b'](t1_scaling.svg)' in first
    assert b'](t1_scaling.json)' in first
    publisher.publish(path, readme)
    assert readme.read_bytes() == first


def test_source_change_blocks_readme_write(report, tmp_path, monkeypatch):
    path = tmp_path / 't1_scaling.json'
    path.write_text(json.dumps(report))
    readme = tmp_path / 'README.md'
    readme.write_text('Existing content')
    monkeypatch.setattr(publisher, 'source_digest', lambda: 'changed')
    with pytest.raises(ValueError, match='source changed'):
        publisher.publish(path, readme)
    assert readme.read_text() == 'Existing content'
