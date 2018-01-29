from collections import OrderedDict
import math
import os
import unittest

import numpy as np
import pystan
import pystan.chains


class TestMcmcChains(unittest.TestCase):
    # REF: stan/src/test/unit/mcmc/chains_test.cpp
    @classmethod
    def setUpClass(cls):
        testdata_path = os.path.join(os.path.dirname(__file__), 'data')
        f1 = os.path.join(testdata_path, 'blocker.1.csv')
        f2 = os.path.join(testdata_path, 'blocker.2.csv')

        # read csv using numpy
        c1 = np.loadtxt(f1, skiprows=41, delimiter=',')[:, 4:]
        c1_colnames = open(f1, 'r').readlines()[36].strip().split(',')[4:]
        np.testing.assert_equal(c1_colnames[0], 'd')
        c2 = np.loadtxt(f2, skiprows=41, delimiter=',')[:, 4:]
        c2_colnames = open(f2, 'r').readlines()[36].strip().split(',')[4:]
        np.testing.assert_equal(c1_colnames, c2_colnames)
        np.testing.assert_equal(len(c1_colnames), c1.shape[1])

        n_samples = len(c1)
        np.testing.assert_equal(n_samples, 1000)

        c1 = OrderedDict((k, v) for k, v in zip(c1_colnames, c1.T))
        c2 = OrderedDict((k, v) for k, v in zip(c2_colnames, c2.T))

        cls.lst = dict(fnames_oi=c1_colnames, samples=[{'chains': c1}, {'chains': c2}],
                       n_save=np.repeat(n_samples, 2), permutation=None,
                       warmup=0, warmup2=[0, 0], chains=2, n_flatnames=len(c1))

    def test_essnrhat(self):
        n_eff = [
            466.099,136.953,1170.390,541.256,
            518.051,589.244,764.813,688.294,
            323.777,502.892,353.823,588.142,
            654.336,480.914,176.978,182.649,
            642.389,470.949,561.947,581.187,
            446.389,397.641,338.511,678.772,
            1442.250,837.956,869.865,951.124,
            619.336,875.805,233.260,786.568,
            910.144,231.582,907.666,747.347,
            720.660,195.195,944.547,767.271,
            723.665,1077.030,470.903,954.924,
            497.338,583.539,697.204,98.421
        ]
        lst = self.lst
        for i in range(4, len(n_eff)):
            ess = pystan.chains.ess(lst, i)
            self.assertAlmostEqual(ess, n_eff[i], delta=.01)

    def test_rhat(self):

        r_hat = [
            1.00718, 1.00473, 0.999203, 1.00061, 1.00378,
            1.01031, 1.00173, 1.0045, 1.00111, 1.00337,
            1.00546, 1.00105, 1.00558, 1.00463, 1.00534,
            1.01244, 1.00174, 1.00718, 1.00186, 1.00554,
            1.00436, 1.00147, 1.01017, 1.00162, 1.00143,
            1.00058, 0.999221, 1.00012, 1.01028, 1.001,
            1.00305, 1.00435, 1.00055, 1.00246, 1.00447,
            1.0048, 1.00209, 1.01159, 1.00202, 1.00077,
            1.0021, 1.00262, 1.00308, 1.00197, 1.00246,
            1.00085, 1.00047, 1.00735
        ]

        lst = self.lst
        for i in range(4, len(r_hat)):
            rhat = pystan.chains.splitrhat(lst, i)
            self.assertAlmostEqual(rhat, r_hat[i], delta=0.001)


class TestRhatZero(unittest.TestCase):

    def test_rhat_zero(self):
        model_code = """
          parameters { real x; }
          transformed parameters { real y; y = 0.0; }
          model {x ~ normal(0, 1);}
          """
        model = pystan.StanModel(model_code=model_code)
        fit = model.sampling()
        rhat = pystan.chains.splitrhat(fit.sim, 1)
        self.assertTrue(math.isnan(rhat))
