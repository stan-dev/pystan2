from collections import OrderedDict
import math
import os
import unittest

import numpy as np
import pystan
import pystan.chains


class TestChains(unittest.TestCase):
    # REF: runit.test.chains.R

    @classmethod
    def setUpClass(cls):
        testdata_path = os.path.join(os.path.dirname(__file__), 'data')
        f1 = os.path.join(testdata_path, 'blocker1.csv')
        f2 = os.path.join(testdata_path, 'blocker2.csv')

        # read csv using numpy
        c1 = np.loadtxt(f1, skiprows=21, delimiter=',')[:, 2:]
        c1_colnames = open(f1, 'r').readlines()[20].strip().split(',')[2:]
        c2 = np.loadtxt(f2, skiprows=21, delimiter=',')[:, 2:]
        c2_colnames = open(f2, 'r').readlines()[20].strip().split(',')[2:]
        np.testing.assert_equal(c1_colnames, c2_colnames)

        n_samples = len(c1)

        c1 = OrderedDict((k, v) for k, v in zip(c1_colnames, c1.T))
        c2 = OrderedDict((k, v) for k, v in zip(c2_colnames, c2.T))

        cls.lst = dict(fnames_oi=c1_colnames, samples=[{'chains': c1}, {'chains': c2}],
                       n_save=np.repeat(n_samples, 2), permutation=None,
                       warmup=0, warmup2=[0, 0], chains=2, n_flatnames=len(c1))

    def test_essnrhat(self):
        lst = self.lst
        ess = pystan.chains.ess(lst, 2)
        self.assertAlmostEqual(ess, 13.0778, delta=1)
        ess2 = pystan.chains.ess(lst, 45)
        self.assertAlmostEqual(ess2, 43.0242, delta=3)

    def test_rhat(self):
        lst = self.lst
        rhat = pystan.chains.splitrhat(lst, 2)
        self.assertAlmostEqual(rhat, 1.187, delta=0.001)
        rhat2 = pystan.chains.splitrhat(lst, 45)
        self.assertAlmostEqual(rhat2, 1.03715, delta=0.001)

    def test_rhat_zero(self):
        model_code = """
          parameters { real x; }
          transformed parameters { real y; y <- 0.0; }
          model {x ~ normal(0, 1);}
          """
        model = pystan.StanModel(model_code=model_code)
        fit = model.sampling()
        rhat = pystan.chains.splitrhat(fit.sim, 1)
        self.assertTrue(math.isnan(rhat))


class TestChainsNormal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        testdata_path = os.path.join(os.path.dirname(__file__), 'data')
        f1 = os.path.join(testdata_path, 'normal1.csv')
        f2 = os.path.join(testdata_path, 'normal2.csv')

        # samples drawn using Stan 2.11.0
        # read csv using numpy
        c1 = np.loadtxt(f1, skiprows=42, delimiter=',')[:, 2:]
        c1_colnames = open(f1, 'r').readlines()[37].strip().split(',')[2:]
        c2 = np.loadtxt(f2, skiprows=42, delimiter=',')[:, 2:]
        c2_colnames = open(f2, 'r').readlines()[37].strip().split(',')[2:]
        np.testing.assert_equal(c1_colnames, c2_colnames)

        assert len(c1) == len(c2) == 40, (len(c1), len(c2))
        y_index = c1_colnames.index('y')
        y_mean = np.concatenate([c1[:, y_index], c2[:, y_index]]).mean()
        np.testing.assert_almost_equal(y_mean, -0.05038095325)

        n_samples = len(c1)

        c1 = OrderedDict((k, v) for k, v in zip(c1_colnames, c1.T))
        c2 = OrderedDict((k, v) for k, v in zip(c2_colnames, c2.T))

        cls.lst = dict(fnames_oi=c1_colnames, samples=[{'chains': c1}, {'chains': c2}],
                       n_save=np.repeat(n_samples, 2), permutation=None,
                       warmup=0, warmup2=[0, 0], chains=2, n_flatnames=len(c1))


    def test_essnrhat(self):
        lst = self.lst
        param_index = lst['fnames_oi'].index('y')
        assert param_index == 5
        ess = pystan.chains.ess(lst, param_index)
        self.assertAlmostEqual(ess, 53.7651, delta=1)

    def test_rhat(self):
        lst = self.lst
        param_index = lst['fnames_oi'].index('y')
        rhat = pystan.chains.splitrhat(lst, param_index)
        self.assertAlmostEqual(rhat, 9.9254714e-01, delta=0.000001)
