import logging
import unittest

import numpy as np

import pystan
from pystan.tests.helper import get_model

class TestOptim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        stdnorm = """
        data {
        int N;
        real y[N];
        }

        parameters {
        real mu;
        real<lower=0> sigma;
        }

        model {
        y ~ normal(mu, sigma);
        }
        """

        N = 30
        np.random.seed(1)
        y = np.random.normal(size=N)
        cls.dat = {'N': N, 'y': y}
        cls.sm = get_model("normal_mu_sigma_model", stdnorm, verbose=True)
        #cls.sm = pystan.StanModel(model_code=stdnorm, verbose=True)

    def test_optim_stdnorm(self):
        optim = self.sm.optimizing(data=self.dat)
        self.assertTrue(-1 < optim['mu'] < 1)
        self.assertTrue(0 < optim['sigma'] < 2)

    def test_optim_stdnorm_from_file(self):
        sm = self.sm
        dat = self.dat
        dump_fn = 'optim_data.Rdump'
        pystan.misc.stan_rdump(dat, dump_fn)
        data_from_file = pystan.misc.read_rdump(dump_fn)
        optim = sm.optimizing(data=data_from_file, algorithm='BFGS')

    def test_optim_stdnorm_bfgs(self):
        sm = self.sm
        dat = self.dat
        optim = sm.optimizing(data=dat, algorithm='BFGS')
        self.assertTrue(-1 < optim['mu'] < 1)
        self.assertTrue(0 < optim['sigma'] < 2)
        optim2 = sm.optimizing(data=dat, algorithm='BFGS',
                               sample_file='opt.csv', init_alpha=0.02,
                               tol_obj=1e-7, tol_grad=1e-9, tol_param=1e-7)
        self.assertTrue(-1 < optim['mu'] < 1)
        self.assertTrue(0 < optim['sigma'] < 2)
        print(optim2)

    def test_optim_stdnorm_lbfgs(self):
        sm = self.sm
        optim = sm.optimizing(data=self.dat, algorithm='LBFGS', seed=5, init_alpha=0.02,
                              tol_obj=1e-7, tol_grad=1e-9, tol_param=1e-7)
        self.assertTrue(-3 < optim['mu'] < 3)
        self.assertTrue(0 < optim['sigma'] < 5)
        optim = sm.optimizing(data=self.dat, algorithm='LBFGS', seed=5, init_alpha=0.02,
                              tol_obj=1e-7, tol_grad=1e-9, tol_param=1e-7, as_vector=False)
        self.assertTrue(-3 < optim['par']['mu'] < 3)
        self.assertTrue(0 < optim['par']['sigma'] < 5)
        optim = sm.optimizing(data=self.dat, algorithm='LBFGS', seed=5, history_size=10)
        self.assertTrue(-3 < optim['mu'] < 3)
        self.assertTrue(0 < optim['sigma'] < 5)
