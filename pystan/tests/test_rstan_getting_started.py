import tempfile
import unittest

import numpy as np

import pystan
from pystan.tests.helper import get_model


def validate_data(fit):
    la = fit.extract(permuted=True)  # return a dictionary of arrays
    mu, tau, eta, theta = la['mu'], la['tau'], la['eta'], la['theta']
    np.testing.assert_equal(mu.shape, (2000,))
    np.testing.assert_equal(tau.shape, (2000,))
    np.testing.assert_equal(eta.shape, (2000, 8))
    np.testing.assert_equal(theta.shape, (2000, 8))
    assert -1 < np.mean(mu) < 17
    assert 0 < np.mean(tau) < 17
    assert all(-3 < np.mean(eta, axis=0))
    assert all(np.mean(eta, axis=0) < 3)
    assert all(-15 < np.mean(theta, axis=0))
    assert all(np.mean(theta, axis=0) < 30)

    # return an array of three dimensions: iterations, chains, parameters
    a = fit.extract(permuted=False)
    np.testing.assert_equal(a.shape, (500, 4, 19))


class TestRStanGettingStarted(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.schools_code = schools_code = """
        data {
            int<lower=0> J; // number of schools
            real y[J]; // estimated treatment effects
            real<lower=0> sigma[J]; // s.e. of effect estimates
        }
        parameters {
            real mu;
            real<lower=0> tau;
            real eta[J];
        }
        transformed parameters {
            real theta[J];
            for (j in 1:J)
            theta[j] = mu + tau * eta[j];
        }
        model {
            eta ~ normal(0, 1);
            y ~ normal(theta, sigma);
        }
        """

        cls.schools_dat = schools_dat = {
            'J': 8,
            'y': [28,  8, -3,  7, -1,  1, 18, 12],
            'sigma': [15, 10, 16, 11,  9, 11, 10, 18]
        }

        cls.sm = sm = get_model("schools_model", schools_code)
        #cls.sm = sm = pystan.StanModel(model_code=schools_code)
        cls.fit = sm.sampling(data=schools_dat, iter=1000, chains=4)

    def test_stan(self):
        fit = self.fit
        validate_data(fit)

    def test_stan_file(self):
        schools_code = self.schools_code
        schools_dat = self.schools_dat
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(schools_code.encode('utf-8'))
        fit = pystan.stan(file=f.name, data=schools_dat, iter=1000, chains=4)
        validate_data(fit)

    def test_stan_reuse_fit(self):
        fit1 = self.fit
        schools_dat = self.schools_dat
        fit = pystan.stan(fit=fit1, data=schools_dat, iter=1000, chains=4)
        validate_data(fit)

    def test_sampling_parallel(self):
        sm = self.sm
        schools_dat = self.schools_dat

        fit = sm.sampling(data=schools_dat, iter=1000, chains=4, n_jobs=-1)
        validate_data(fit)

        # n_jobs specified explicitly
        fit = sm.sampling(data=schools_dat, iter=1000, chains=4, n_jobs=4)
        validate_data(fit)
