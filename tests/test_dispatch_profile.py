"""Tests for the stored population-evaluation route calibration."""

import json

import _dispatch_profile as profile


def _entry(samples, decision, rules=10, features=4, population=20):
    return {'samples': samples, 'rules': rules, 'features': features,
            'population': population, 'decision': decision}


def test_default_path_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(profile.PROFILE_ENV, str(tmp_path / 'custom.json'))
    assert profile.default_path() == str(tmp_path / 'custom.json')

    monkeypatch.delenv(profile.PROFILE_ENV)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    assert profile.default_path() == str(tmp_path / 'ex-fuzzy' / 'dispatch_profile.json')


def test_profiles_round_trip_and_unusable_ones_are_ignored(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # A bare file name has no directory to create.
    assert profile.save([], 'profile.json') == 'profile.json'
    assert profile.load('profile.json') is None

    profile.save([_entry(100, 'batch')], 'profile.json')
    assert profile.load('profile.json') == [_entry(100, 'batch')]
    assert profile.decision_for(100, 10, 4, 20, path='profile.json') == 'batch'
    assert profile.decision_for(100, 10, 4, 20, path='missing.json') is None


def test_profiles_from_other_machines_are_ignored(tmp_path):
    path = tmp_path / 'nested' / 'profile.json'
    profile.save([_entry(100, 'batch')], str(path))
    assert profile.load(str(path)) == [_entry(100, 'batch')]

    payload = json.loads(path.read_text())
    payload['fingerprint']['machine'] = 'elsewhere'
    path.write_text(json.dumps(payload))
    assert profile.load(str(path)) is None


def test_lookup_prefers_the_nearest_workload():
    near = _entry(110, 'scalar')
    far = _entry(180, 'batch')
    broken = {'samples': 'many'}

    assert profile.lookup([near, far, broken], 100, 10, 4, 20) == 'scalar'
    assert profile.lookup([far, near], 100, 10, 4, 20) == 'scalar'
    assert profile.lookup([near], 0, 10, 4, 20) is None
    assert profile.lookup([near], 1000, 10, 4, 20) is None
