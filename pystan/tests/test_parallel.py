import unittest

import numpy as np

import pystan


def validate_data(fit):
    la = fit.extract(permuted=True)  # return a dictionary of arrays
    mu, tau, eta, theta = la['mu'], la['tau'], la['eta'], la['theta']
    assert mu.shape == (2000,)
    assert tau.shape == (2000,)
    assert eta.shape == (2000, 8)
    assert theta.shape == (2000, 8)
    assert -1 < np.mean(mu) < 17
    assert 0 < np.mean(tau) < 17
    assert all(-3 < np.mean(eta, axis=0))
    assert all(np.mean(eta, axis=0) < 3)
    assert all(-15 < np.mean(theta, axis=0))
    assert all(np.mean(theta, axis=0) < 30)

    ## return an array of three dimensions: iterations, chains, parameters
    a = fit.extract(permuted=False)
    assert a.shape == (500, 4, 19)


class Test8SchoolsParallel(unittest.TestCase):

    schools_code = """
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
        theta[j] <- mu + tau * eta[j];
    }
    model {
        eta ~ normal(0, 1);
        y ~ normal(theta, sigma);
    }
    """

    schools_dat = {'J': 8,
                   'y': [28,  8, -3,  7, -1,  1, 18, 12],
                   'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

    def test_stan_parallel(self):
        schools_code = self.schools_code
        schools_dat = self.schools_dat
        fit = pystan.stan(model_code=schools_code, data=schools_dat,
                          iter=1000, chains=4, n_jobs=-1)
        validate_data(fit)

    def test_sampling_parallel(self):
        schools_code = self.schools_code
        schools_dat = self.schools_dat
        sm = pystan.StanModel(model_code=schools_code)
        fit = sm.sampling(data=schools_dat, iter=1000, chains=4, n_jobs=4)
        validate_data(fit)
