import os
import tempfile
from collections import OrderedDict

import numpy as np

from pystan.constants import MAX_UINT
from pystan import misc


def test_pars_total_indexes():
    pars_oi = ['mu', 'tau', 'eta', 'theta', 'lp__']
    dims_oi = [[], [], [8], [8], []]
    fnames_oi = ['mu', 'tau', 'eta[1]', 'eta[2]', 'eta[3]', 'eta[4]', 'eta[5]',
                 'eta[6]', 'eta[7]', 'eta[8]', 'theta[1]', 'theta[2]',
                 'theta[3]', 'theta[4]', 'theta[5]', 'theta[6]', 'theta[7]',
                 'theta[8]', 'lp__']
    pars = ['mu', 'tau', 'eta', 'theta', 'lp__']
    rslt = misc._pars_total_indexes(pars_oi, dims_oi, fnames_oi, pars)
    assert rslt['mu'] == (0,)
    assert rslt['tau'] == (1,)
    assert rslt['eta'] == (2, 3, 4, 5, 6, 7, 8, 9)
    assert rslt['theta'] == (10, 11, 12, 13, 14, 15, 16, 17)
    assert rslt['lp__'] == (18,)


def test_par_vector2dict():
    v = [0,  1, -1, -1,  0,  1, -1, -2]
    pars = ['alpha', 'beta']
    dims = [[2, 3], [2]]
    rslt = misc._par_vector2dict(v, pars, dims)
    rslt['alpha'] = np.array([0,  1, -1, -1, 0,  1]).reshape(dims[0])
    rslt['beta'] = np.array([-1, -2])


def is_valid_seed(seed):
    return isinstance(seed, int) and seed >= 0 and seed <= MAX_UINT


def test_check_seed():
    assert misc._check_seed('10') == 10
    assert misc._check_seed(10) == 10
    assert misc._check_seed(10.5) == 10
    assert is_valid_seed(misc._check_seed(np.random.RandomState()))
    assert is_valid_seed(misc._check_seed(-1))
    assert is_valid_seed(misc._check_seed(MAX_UINT + 1))


def test_stan_rdump():
    data = OrderedDict(x=1, y=0, z=[1, 2, 3], Phi=np.array([[1, 2], [3, 4]]))
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_rdump.rdump')
    misc.stan_rdump(data, filename)
    data_recovered = misc.read_rdump(filename)
    np.testing.assert_equal(data, data_recovered)

    data = OrderedDict(x=True, y=np.array([True, False]))
    data_expected = OrderedDict(x=1, y=np.array([1, 0]))
    misc.stan_rdump(data, filename)
    data_recovered = misc.read_rdump(filename)
    np.testing.assert_equal(data_recovered, data_expected)

    data = OrderedDict(x='foo')
    np.testing.assert_raises(ValueError, misc.stan_rdump, data, filename)

    data = OrderedDict(new=3)
    np.testing.assert_raises(ValueError, misc.stan_rdump, data, filename)


def test_rstan_read_rdump():
    a = np.array([1, 3, 5])
    b = np.arange(10).reshape((-1, 2))
    c = np.arange(18).reshape([2, 3, 3])
    data = dict(**locals())
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_rdump.rdump')
    misc.stan_rdump(data, filename)
    d = misc.read_rdump(filename)
    np.testing.assert_equal(d['a'], a)
    np.testing.assert_equal(d['b'], b)
    np.testing.assert_equal(d['c'], c)


def test_remove_empty_pars():
    pars = ('alpha', 'beta', 'gamma', 'eta', 'xi')
    dims = [[], (2,), (2, 4), 0, (2, 0)]
    # NOTE: np.prod([]) -> 1
    np.testing.assert_equal(misc._remove_empty_pars(pars[0:1], pars, dims), pars[0:1])
    np.testing.assert_equal(misc._remove_empty_pars(pars[0:3], pars, dims), pars[0:3])
    np.testing.assert_equal(misc._remove_empty_pars(pars[0:4], pars, dims), pars[0:3])
    np.testing.assert_equal(misc._remove_empty_pars(['beta[1]'], pars, dims), ['beta[1]'])
    np.testing.assert_equal(misc._remove_empty_pars(['eta'], pars, dims), [])
