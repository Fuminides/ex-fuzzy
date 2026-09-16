"""Type-1 rule-base inference equals the per-sample centroid up to rounding."""
import numpy as np

from ex_fuzzy import centroid, fuzzy_sets as fs, rules, utils


def _regression_base(seed):
    rng = np.random.default_rng(seed)
    X = rng.random((300, 3))
    antecedents = utils.construct_partitions(X, fs.FUZZY_SETS.t1, detect_categorical=False)
    consequent = fs.fuzzyVariable('y', [fs.FS('low', [0, 0, 0.3, 0.5], [0, 1]),
                                        fs.FS('mid', [0.3, 0.5, 0.5, 0.7], [0, 1]),
                                        fs.FS('high', [0.5, 0.7, 1, 1], [0, 1])])
    rule_list = [rules.RuleSimple(rng.integers(-1, 3, 3), int(rng.integers(0, 3))) for _ in range(8)]
    return rules.RuleBaseT1(antecedents, rule_list, consequent), X


def test_inference_matches_per_sample_centroids():
    base, X = _regression_base(3)
    firing = base.compute_rule_antecedent_memberships(X)
    expected = np.array([centroid.center_of_masses(base.consequent_centroids_rules, row) for row in firing])

    actual = base.inference(X)
    assert actual.shape == (len(X),)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0)
    np.testing.assert_array_equal(base.forward(X), actual)
