import logging
import unittest

import numpy as np

import pystan


class TestOptim(unittest.TestCase):
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
    dat = {'N': N, 'y': y}
    logging.info("mean(y)={} and sd(y)={}".format(np.mean(y),
                                                  np.std(y, ddof=1)))
    sm = pystan.StanModel(model_code=stdnorm)

    def test_optim_stdnorm(self):
        optim = self.sm.optimizing(data=self.dat)
        print(optim)
        self.assertTrue(-1 < optim['mu'] < 1)
        self.assertTrue(0 < optim['sigma'] < 2)

    def test_optim_stdnorm_from_file(self):
        sm = self.sm
        dat = self.dat
        dump_fn = 'optim_data.Rdump'
        pystan.misc.stan_rdump(dat, dump_fn)
        data_from_file = pystan.misc.read_rdump(dump_fn)
        optim = sm.optimizing(data=data_from_file, algorithm='BFGS')
        print(optim)

    def test_optim_stdnorm_bfgs(self):
        sm = self.sm
        dat = self.dat
        optim = sm.optimizing(data=dat, algorithm='BFGS')
        print(optim)
        self.assertTrue(-1 < optim['mu'] < 1)
        self.assertTrue(0 < optim['sigma'] < 2)
        optim2 = sm.optimizing(data=dat, algorithm='BFGS',
                               sample_file='opt.csv', init_alpha=0.02,
                               tol_obj=1e-7, tol_grad=1e-9, tol_param=1e-7)
        print(optim2)

    def test_optim_stdnorm_lbfgs(self):
        sm = self.sm
        optim = sm.optimizing(data=self.dat, algorithm='LBFGS', seed=5, init_alpha=0.02,
                              tol_obj=1e-7, tol_grad=1e-9, tol_param=1e-7)
        self.assertTrue(-3 < optim['mu'] < 3)
        self.assertTrue(0 < optim['sigma'] < 5)
        optim = sm.optimizing(data=self.dat, algorithm='LBFGS', seed=5, init_alpha=0.02,
                              tol_obj=1e-7, tol_grad=1e-9, tol_param=1e-7, as_vector=False)
