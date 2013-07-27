import logging
import unittest

import numpy as np

from pystan import StanModel


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
    y = np.random.normal(size=N)
    dat = {'N': N, 'y': y}
    logging.info("mean(y)={} and sd(y)={}".format(np.mean(y),
                                                  np.std(y, ddof=1)))
    sm = StanModel(model_code=stdnorm)

    def test_optim_stdnorm(self):
        optim = self.sm.optimizing(data=self.dat)
        print(optim)
        self.assertTrue(-1 < optim['par']['mu'] < 1)
        self.assertTrue(0 < optim['par']['sigma'] < 2)

    # FIXME: to implement
    # def test_optim_stdnorm_from_file(self):
    #     dump(c("N", "y"), file = 'optim.data.R')
    #     optim = sm.optimizing(file='optim.data.R', method='BFGS')

    def test_optim_stdnorm_bfgs(self):
        optim = self.sm.optimizing(data=self.dat, method='BFGS')
        print(optim)
        self.assertTrue(-1 < optim['par']['mu'] < 1)
        self.assertTrue(0 < optim['par']['sigma'] < 2)

    def test_optim_stdnorm_nesterov(self):
        optim = self.sm.optimizing(data=self.dat, method='Nesterov')
        self.assertTrue(-1 < optim['par']['mu'] < 1)
        self.assertTrue(0 < optim['par']['sigma'] < 2)
