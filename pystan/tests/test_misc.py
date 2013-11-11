import numpy as np

from pystan.constants import MAX_UINT
from pystan.misc import _pars_total_indexes, _par_vector2dict, _check_seed


def test_pars_total_indexes():
    pars_oi = ['mu', 'tau', 'eta', 'theta', 'lp__']
    dims_oi = [[], [], [8], [8], []]
    fnames_oi = ['mu', 'tau', 'eta[1]', 'eta[2]', 'eta[3]', 'eta[4]', 'eta[5]',
                 'eta[6]', 'eta[7]', 'eta[8]', 'theta[1]', 'theta[2]',
                 'theta[3]', 'theta[4]', 'theta[5]', 'theta[6]', 'theta[7]',
                 'theta[8]', 'lp__']
    pars = ['mu', 'tau', 'eta', 'theta', 'lp__']
    rslt = _pars_total_indexes(pars_oi, dims_oi, fnames_oi, pars)
    assert rslt['mu'] == (0,)
    assert rslt['tau'] == (1,)
    assert rslt['eta'] == (2, 3, 4, 5, 6, 7, 8, 9)
    assert rslt['theta'] == (10, 11, 12, 13, 14, 15, 16, 17)
    assert rslt['lp__'] == (18,)


def test_par_vector2dict():
    v = [0,  1, -1, -1,  0,  1, -1, -2]
    pars = ['alpha', 'beta']
    dims = [[2, 3], [2]]
    rslt = _par_vector2dict(v, pars, dims)
    rslt['alpha'] = np.array([0,  1, -1, -1, 0,  1]).reshape(dims[0])
    rslt['beta'] = np.array([-1, -2])


def is_valid_seed(seed):
    return isinstance(seed, int) and seed >= 0 and seed <= MAX_UINT


def test_check_seed():
    assert _check_seed('10') == 10
    assert _check_seed(10) == 10
    assert _check_seed(10.5) == 10
    assert is_valid_seed(_check_seed(np.random.RandomState()))
    assert is_valid_seed(_check_seed(-1))
    assert is_valid_seed(_check_seed(MAX_UINT + 1))
