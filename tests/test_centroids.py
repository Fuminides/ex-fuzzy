import numpy as np
import math

import pytest

import sys
sys.path.append('../ex_fuzzy/')

import ex_fuzzy as ex_fuzzy

centroid = ex_fuzzy.centroid


def _brute_force_iv_centroid(z, memberships):
    '''Centroid bounds of an IV fuzzy set by trying every switch point.'''
    lower, upper = memberships[:, 0], memberships[:, 1]
    left, right = [], []
    for k in range(len(z) + 1):
        weights_left = np.r_[upper[:k], lower[k:]]
        weights_right = np.r_[lower[:k], upper[k:]]
        if weights_left.sum() > 0:
            left.append(z @ weights_left / weights_left.sum())
        if weights_right.sum() > 0:
            right.append(z @ weights_right / weights_right.sum())
    return min(left), max(right)


def test_t2_centroid():
    '''
    Tests that fuzzy t2 (iv) centroids compute correctly.

    The set is symmetric around 0.5, so its centroid is an interval centred at 0.5
    whose width comes from the footprint of uncertainty.
    '''
    trial_fs = ex_fuzzy.fuzzy_sets.IVFS('trial', [0.35, 0.4, 0.6, 0.65], [0.25 ,0.4, 0.6, 0.75], [0,1], lower_height=0.8)
    z = np.arange(0, 1, 0.001)
    left, right = ex_fuzzy.centroid.compute_centroid_iv(z, trial_fs(z))
    assert left < 0.5 < right
    assert math.isclose(left + right, 1.0, abs_tol=0.01), 'T2 centroid is not symmetric'
    np.testing.assert_allclose((left, right), _brute_force_iv_centroid(z, trial_fs(z)))


def test_t2_centroid_centroids():
    '''
    Tests that the left and right components of a fuzzy t2 (iv) centroid are computed correctly.
    '''
    trial_fs = ex_fuzzy.fuzzy_sets.IVFS('trial', [0.35, 0.4, 0.6, 0.65], [0.25 ,0.4, 0.6, 0.75], [0,1], lower_height=0.8)
    z = np.arange(0, 1, 0.001)
    left, right = _brute_force_iv_centroid(z, trial_fs(z))
    assert math.isclose(ex_fuzzy.centroid.compute_centroid_t2_r(z, trial_fs(z)), right), 'T2 centroid right not correctly computed'
    assert math.isclose(ex_fuzzy.centroid.compute_centroid_t2_l(z, trial_fs(z)), left), 'T2 centroid left not correctly computed'


def test_center_of_masses_is_the_weighted_mean():
    assert centroid.center_of_masses(np.array([1.0, 3.0]), np.array([1.0, 3.0])) == pytest.approx(2.5)


def test_iv_centroid_matches_exhaustive_switch_point_search():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = rng.integers(2, 20)
        z = np.sort(rng.uniform(-3, 5, n))
        upper = rng.uniform(0.05, 1, n)
        memberships = np.stack([upper * rng.uniform(0, 1, n), upper], axis=1)

        expected = _brute_force_iv_centroid(z, memberships)
        np.testing.assert_allclose(centroid.compute_centroid_iv(z, memberships.copy()), expected)


def test_iv_centroid_of_a_wide_footprint_is_an_interval():
    z = np.linspace(0, 1, 11)
    upper = np.clip(1 - np.abs(z - 0.5) * 2, 0, 1)
    memberships = np.stack([upper * 0.5, upper], axis=1)

    left, right = centroid.compute_centroid_iv(z, memberships)
    assert left < 0.5 < right
    assert left == pytest.approx(0.44285714)
    assert right == pytest.approx(0.55714286)


def test_iv_centroid_when_the_centroid_is_left_of_every_point():
    # All the mass sits on the first point, so no referencial value lies below the centroid estimate.
    z = np.array([1.0, 2.0])
    memberships = np.array([[1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(centroid.compute_centroid_iv(z, memberships.copy()), [1.0, 1.0])


def test_iv_centroid_of_an_empty_set_is_undefined():
    z = np.linspace(0, 1, 5)
    assert np.isnan(centroid.compute_centroid_t2_l(z, np.zeros((5, 2))))
    assert np.isnan(centroid.compute_centroid_t2_r(z, np.zeros((5, 2))))


def test_consequent_centroid_matches_exhaustive_switch_point_search():
    antecedents = np.array([[0.2, 0.5], [0.1, 0.9], [0.0, 0.3]])
    centroids = np.array([[0.1, 0.2], [0.4, 0.6], [0.8, 0.9]])

    left, right = centroid.consequent_centroid(antecedents.copy(), centroids)
    assert left == pytest.approx(_brute_force_iv_centroid(centroids[:, 0], antecedents)[0])
    assert right == pytest.approx(_brute_force_iv_centroid(centroids[:, 1], antecedents)[1])


def test_consequent_centroid_without_firing_rules_is_zero():
    centroids = np.array([[0.1, 0.2], [0.4, 0.6], [0.8, 0.9]])
    np.testing.assert_array_equal(centroid.consequent_centroid(np.zeros((3, 2)), centroids), [0.0, 0.0])


def test_consequent_centroid_below_every_rule_centroid():
    # The first estimate lies below every consequent centroid, which exercises the empty switch point search.
    antecedents = np.array([[1.0, 1.0], [0.0, 0.0]])
    centroids = np.array([[0.5, 0.5], [0.9, 0.9]])
    np.testing.assert_allclose(centroid.consequent_centroid(antecedents.copy(), centroids), [0.5, 0.5])
