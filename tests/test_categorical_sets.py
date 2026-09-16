"""Categorical fuzzy sets carry no numerical domain."""
import numpy as np

from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils


def test_categorical_variables_report_no_domain():
    variable = utils.construct_crisp_categorical_partition(
        np.array(['a', 'b', 'a']), 'colour', fs.FUZZY_SETS.t1)
    assert variable.domain() is None
    assert all(fuzzy_set.domain is None and fuzzy_set.membership_parameters is None
               for fuzzy_set in variable)
    np.testing.assert_array_equal(
        variable.compute_memberships(np.array(['a', 'b', 'c'])), [[1, 0, 0], [0, 1, 0]])

    interval = utils.construct_crisp_categorical_partition(np.array([1, 2]), 'code', fs.FUZZY_SETS.t2)
    assert interval.domain() is None
    assert interval.compute_memberships(np.array([2, 3])).shape == (2, 2, 2)


def test_numerical_variables_still_clip_to_their_domain():
    variable = fs.fuzzyVariable('x', [fs.FS('low', [0, 0, 0.5, 1], [0, 1]),
                                      fs.FS('high', [0, 0.5, 1, 1], [0, 1])])
    np.testing.assert_array_equal(variable.compute_memberships(np.array([-5.0, 5.0])),
                                  variable.compute_memberships(np.array([0.0, 1.0])))
