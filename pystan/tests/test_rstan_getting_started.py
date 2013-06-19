import numpy as np

import pystan


def test_rstan_getting_started():
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

    fit = pystan.stan(model_code=schools_code, data=schools_dat,
                      iter=1000, chains=4)

    la = fit.extract(permuted=True)  # return a dictionary of arrays
    mu = la['mu']
    tau = la['tau']
    eta = la['eta']
    theta = la['theta']
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
    print(a.shape)
    assert a.shape == (500, 4, 19)
