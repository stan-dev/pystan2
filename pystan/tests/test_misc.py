from collections import OrderedDict
import os
import tempfile
import unittest

import numpy as np

from pystan import misc
from pystan._compat import PY2
from pystan.constants import MAX_UINT


def is_valid_seed(seed):
    return isinstance(seed, int) and seed >= 0 and seed <= MAX_UINT


class TestMisc(unittest.TestCase):

    def test_pars_total_indexes(self):
        pars_oi = ['mu', 'tau', 'eta', 'theta', 'lp__']
        dims_oi = [[], [], [8], [8], []]
        fnames_oi = ['mu', 'tau', 'eta[1]', 'eta[2]', 'eta[3]', 'eta[4]', 'eta[5]',
                     'eta[6]', 'eta[7]', 'eta[8]', 'theta[1]', 'theta[2]',
                     'theta[3]', 'theta[4]', 'theta[5]', 'theta[6]', 'theta[7]',
                     'theta[8]', 'lp__']
        pars = ['mu', 'tau', 'eta', 'theta', 'lp__']
        rslt = misc._pars_total_indexes(pars_oi, dims_oi, fnames_oi, pars)
        self.assertEqual(rslt['mu'], (0,))
        self.assertEqual(rslt['tau'], (1,))
        self.assertEqual(rslt['eta'], (2, 3, 4, 5, 6, 7, 8, 9))
        self.assertEqual(rslt['theta'], (10, 11, 12, 13, 14, 15, 16, 17))
        self.assertEqual(rslt['lp__'], (18,))

    def test_par_vector2dict(self):
        v = [0, 1, -1, -1, 0, 1, -1, -2]
        pars = ['alpha', 'beta']
        dims = [[2, 3], [2]]
        rslt = misc._par_vector2dict(v, pars, dims)
        np.testing.assert_equal(rslt['alpha'], np.array([0, 1, -1, -1, 0, 1]).reshape(dims[0], order='F'))
        np.testing.assert_equal(rslt['beta'], np.array([-1, -2]))

    def test_check_seed(self):
        self.assertEqual(misc._check_seed('10'), 10)
        self.assertEqual(misc._check_seed(10), 10)
        self.assertEqual(misc._check_seed(10.5), 10)
        self.assertTrue(is_valid_seed(misc._check_seed(np.random.RandomState())))
        self.assertTrue(is_valid_seed(misc._check_seed(-1)))
        self.assertTrue(is_valid_seed(misc._check_seed(MAX_UINT + 1)))

    def test_stan_rdump(self):
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

    def test_stan_rdump_array(self):
        y = np.asarray([[1, 3, 5], [2, 4, 6]])
        data = {'y': y}
        expected = 'y <-\nstructure(c(1, 2, 3, 4, 5, 6), .Dim = c(2, 3))\n'
        np.testing.assert_equal(misc._dict_to_rdump(data), expected)

    def test_stan_read_rdump_array(self):
        """
        For reference:
        > structure(c(1, 2, 3, 4, 5, 6), .Dim = c(2, 3))
            [,1] [,2] [,3]
        [1,]    1    3    5
        [2,]    2    4    6
        """
        # make this into some sort of stream
        rdump = 'y <-\nstructure(c(1, 2, 3, 4, 5, 6), .Dim = c(2, 3))\n'
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_rdump.rdump')
        with open(filename, 'w') as f:
            f.write(rdump)
        d = misc.read_rdump(filename)
        y = np.asarray([[1, 3, 5], [2, 4, 6]])
        np.testing.assert_equal(d['y'], y)

    def test_rstan_read_rdump(self):
        a = np.array([1, 3, 5])
        b = np.arange(10).reshape((-1, 2))
        c = np.arange(18).reshape([2, 3, 3])
        data = dict(a=a, b=b, c=c)
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_rdump.rdump')
        misc.stan_rdump(data, filename)
        d = misc.read_rdump(filename)
        np.testing.assert_equal(d['a'], a)
        np.testing.assert_equal(d['b'], b)
        np.testing.assert_equal(d['c'], c)

    def test_remove_empty_pars(self):
        pars = ('alpha', 'beta', 'gamma', 'eta', 'xi')
        dims = [[], (2,), (2, 4), 0, (2, 0)]
        # NOTE: np.prod([]) -> 1
        np.testing.assert_equal(misc._remove_empty_pars(pars[0:1], pars, dims), pars[0:1])
        np.testing.assert_equal(misc._remove_empty_pars(pars[0:3], pars, dims), pars[0:3])
        np.testing.assert_equal(misc._remove_empty_pars(pars[0:4], pars, dims), pars[0:3])
        np.testing.assert_equal(misc._remove_empty_pars(['beta[1]'], pars, dims), ['beta[1]'])
        np.testing.assert_equal(misc._remove_empty_pars(['eta'], pars, dims), [])

    def test_format_number_si(self):
        np.testing.assert_equal(misc._format_number_si(-12345, 2), '-1.2e4')
        np.testing.assert_equal(misc._format_number_si(123456, 2), '1.2e5')
        np.testing.assert_equal(misc._format_number_si(-123456, 2), '-1.2e5')
        np.testing.assert_equal(misc._format_number_si(-123456.393, 2), '-1.2e5')
        np.testing.assert_equal(misc._format_number_si(1234567, 2), '1.2e6')
        np.testing.assert_equal(misc._format_number_si(0.12, 2), '1.2e-1')
        np.testing.assert_equal(misc._format_number_si(3.32, 2), '3.3e0')

    def test_format_number(self):
        np.testing.assert_equal(misc._format_number(-1234, 2, 6), '-1234')
        np.testing.assert_equal(misc._format_number(-12345, 2, 6), '-12345')
        np.testing.assert_equal(misc._format_number(123456, 2, 6), '123456')
        np.testing.assert_equal(misc._format_number(-123456, 2, 6), '-1.2e5')
        np.testing.assert_equal(misc._format_number(1234567, 2, 6), '1.2e6')
        np.testing.assert_equal(misc._format_number(0.12, 2, 6), '0.12')
        np.testing.assert_equal(misc._format_number(3.32, 2, 6), '3.32')
        np.testing.assert_equal(misc._format_number(3.3, 2, 6), '3.3')
        np.testing.assert_equal(misc._format_number(3.32, 2, 6), '3.32')
        np.testing.assert_equal(misc._format_number(3.323945, 2, 6), '3.32')
        np.testing.assert_equal(misc._format_number(-0.0003434, 2, 6), '-3.4e-4')
        np.testing.assert_equal(misc._format_number(1654.25, 2, 6), '1654.2')
        np.testing.assert_equal(misc._format_number(-1654.25, 2, 6), '-1654')
        np.testing.assert_equal(misc._format_number(-165.25, 2, 6), '-165.2')
        np.testing.assert_equal(misc._format_number(9.94598693e-01, 2, 6), '0.99')
        np.testing.assert_equal(misc._format_number(-9.94598693e-01, 2, 6), '-0.99')

    def test_array_to_table(self):
        rownames = np.array(['alpha', 'beta[0]', 'beta[1]', 'beta[2]', 'beta[3]', 'sigma', 'lp__'])
        colnames = ('mean', 'se_mean', 'sd', '2.5%', '25%', '50%', '75%', '97.5%', 'n_eff', 'Rhat')
        arr = np.array([[-1655,   625,  1654, -4378, -3485,  -927,   -27,    -9,     7, 2],
                        [  -47,    59,   330,  -770,   -69,    -2,     2,  1060,    31, 1],
                        [  -18,    70,   273,  -776,   -30,     0,    35,   608,    15, 1],
                        [    8,    38,   230,  -549,   -26,     0,    17,   604,    36, 1],
                        [   23,    32,   303,  -748,   -34,     0,    49,   751,    92, 1],
                        [  405,   188,   974,     0,     4,    73,   458,  2203,    27, 1],
                        [  -13,     3,     8,   -23,   -20,   -15,    -5,     1,     8, 2]])
        n_digits = 2
        result = misc._array_to_table(arr, rownames, colnames, n_digits)
        desired = "alpha    -1655     625   1654  -4378  -3485   -927    -27     -9      7      2"
        desired_py2 = "alpha    -1655   625.0 1654.0  -4378  -3485 -927.0  -27.0   -9.0    7.0    2.0"
        # round() behaves differently in Python 3
        if PY2:
            np.testing.assert_equal(result.split('\n')[1], desired_py2)
        else:
            np.testing.assert_equal(result.split('\n')[1], desired)
        arr = np.array([[-1655325, 625.25,  1654.25, -4378.25, -3485.25,  -927.25, -27.25,    -9.25,   7.25, 2],
                        [  -47.25,  59.25,   330.25,  -770.25,   -69.25,    -2.25,   2.25,  1060.25,  31.25, 1],
                        [  -18.25,  70.25,   273.25,  -776.25,   -30.25,     0.25,  35.25,   608.25,  15.25, 1],
                        [    8.25,  38.25,   230.25,  -549.25,   -26.25,     0.25,  17.25,   604.25,  36.25, 1],
                        [   23.25,  32.25,   303.25,  -748.25,   -34.25,     0.25,  49.25,   751.25,  92.25, 1],
                        [  405.25, 188.25,   974.25,     0.25,     4.25,    73.25, 458.25,  2203.25,  27.25, 1],
                        [  -13.25,   3.25,     8.25,   -23.25,   -20.25,   -15.25,  -5.25,     1.25,   8.25, 2]])
        result = misc._array_to_table(arr, rownames, colnames, n_digits)
        desired = "alpha   -1.7e6  625.25 1654.2  -4378  -3485 -927.2 -27.25  -9.25      7    2.0"
        desired_py2 = "alpha   -1.7e6  625.25 1654.2  -4378  -3485 -927.2 -27.25  -9.25    7.0    2.0"
        # round() behaves differently in Python 3
        if PY2:
            np.testing.assert_equal(result.split('\n')[1], desired_py2)
        else:
            np.testing.assert_equal(result.split('\n')[1], desired)
