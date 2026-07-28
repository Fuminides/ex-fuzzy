import numpy as np
import math

import sys
sys.path.append('../ex_fuzzy/')

import ex_fuzzy as ex_fuzzy

def test_t2_centroid():
    '''
    Tests that fuzzy t2 (iv) centroids compute correctly.
    '''
    trial_fs = ex_fuzzy.fuzzy_sets.IVFS('trial', [0.35, 0.4, 0.6, 0.65], [0.25 ,0.4, 0.6, 0.75], [0,1], lower_height=0.8)
    z = np.arange(0, 1, 0.001)
    assert math.isclose(ex_fuzzy.centroid.compute_centroid_iv(z, trial_fs(z))[0], 0.5, abs_tol=0.01), 'T2 centroid right not correctly computed'
    assert math.isclose(ex_fuzzy.centroid.compute_centroid_iv(z, trial_fs(z))[1], 0.5, abs_tol=0.01), 'T2 centroid left not correctly computed'


def test_t2_centroid_centroids():
    '''
    Tests that fuzzy t2 (iv) centroid computed from previous centroids is computed correctly.
    '''
    trial_fs = ex_fuzzy.fuzzy_sets.IVFS('trial', [0.35, 0.4, 0.6, 0.65], [0.25 ,0.4, 0.6, 0.75], [0,1], lower_height=0.8)
    z = np.arange(0, 1, 0.001)
    assert math.isclose(ex_fuzzy.centroid.compute_centroid_t2_r(z, trial_fs(z)), 0.5, abs_tol=0.01), 'T2 centroid right not correctly computed'
    assert math.isclose(ex_fuzzy.centroid.compute_centroid_t2_l(z, trial_fs(z)), 0.5, abs_tol=0.01), 'T2 centroid left not correctly computed'


