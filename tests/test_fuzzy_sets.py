import numpy as np
import math

import sys
sys.path.append('./ex_fuzzy/')
sys.path.append('../ex_fuzzy/')
import ex_fuzzy as ex_fuzzy


def test_fuzzy_t1_memberships():
    '''
    Tests that fuzzy t1 memberships compute correctly.
    '''
    trial_fs = ex_fuzzy.fuzzy_sets.FS('trial', [0,0.1, 0.25, 0.5], [0,1])

    assert trial_fs.name == 'trial', 'Name not correctly set'
    assert trial_fs(0.1) == 1, 'Trapezoidal membership fails in first term'
    assert trial_fs(1) == 0, 'Trapezoidal membership fails in last term'
    assert trial_fs(0.51) == 0, 'Trapezoidal memberships does not stop where it shoud'
    assert trial_fs(0.05) == 0.5, 'Trapeziodal membership increasining in non linear way' 


def test_fuzzy_t2_memberships():
    '''
    Tests that fuzzy t2 (iv) memberships compute correctly.
    '''
    trial_fs = ex_fuzzy.fuzzy_sets.IVFS('trial', [0,0.1, 0.25, 0.4], [0, 0.15, 0.25, 0.5], [0,1], lower_height=0.8)

    assert trial_fs.name == 'trial', 'Name not correctly set'
    trial_mfs = trial_fs(0.2)
    assert trial_mfs[0] <= trial_mfs[1], 'Incoherent lower and upper membership behaviour'


def test_fuzzy_gt2_memberships():
    '''
    Tests that fuzzy gt2 memberships compute correctly.
    '''
    trial_fs = {}
    for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]:
        trial_fs[x] = ex_fuzzy.fuzzy_sets.FS('trial', [0.25,0.5, 0.5, 0.75], [0,1])

    trial_fs = ex_fuzzy.fuzzy_sets.GT2('trial', trial_fs, [0, 1], significant_decimals=1, unit_resolution=0.001)
    res = trial_fs(0.3)
    assert math.isclose(np.mean(trial_fs.alpha_reduction(res)), 0.5, abs_tol=0.1), 'GT2 memberships not correctly reduced'



    
if __name__ == '__main__':
    test_fuzzy_t1_memberships()
    test_fuzzy_t2_memberships()
    test_fuzzy_gt2_memberships()