import unittest

import numpy as np

from pystan import StanModel, stan
from pystan._compat import PY2


# REF: rstan/tests/unitTests/runit.test.stanfit.R

class TestStanfit(unittest.TestCase):

    def test_init_zero_exception_inf_grad(self):
        code = """
        parameters {
            real x;
        }
        model {
            lp__ <- 1 / log(x);
        }
        """
        sm = StanModel(model_code=code)
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(RuntimeError, 'Gradient evaluated at the initial value is not finite.'):
            sm.sampling(init='0', iter=1)

    def test_grad_log(self):
        y = np.array([0.70,  -0.16,  0.77, -1.37, -1.99,  1.35, 0.08,
                      0.02,  -1.48, -0.08,  0.34,  0.03, -0.42, 0.87,
                      -1.36,  1.43,  0.80, -0.48, -1.61, -1.27])

        code = '''
        data {
            real y[20];
        }
        parameters {
            real mu;
            real<lower=0> sigma;
        }
        model {
            y ~ normal(mu, sigma);
        }'''

        def log_prob_fun(mu, log_sigma, adjust=True):
            sigma = np.exp(log_sigma)
            lp = -1 * np.sum((y - mu)**2) / (2 * (sigma**2)) - len(y) * np.log(sigma)
            if adjust:
                lp = lp + np.log(sigma)
            return lp

        def log_prob_grad_fun(mu, log_sigma, adjust=True):
            sigma = np.exp(log_sigma)
            g_lsigma = np.sum((y - mu)**2) * sigma**(-2) - len(y)
            if adjust:
                g_lsigma = g_lsigma + 1
            g_mu = np.sum(y - mu) * sigma**(-2)
            return (g_mu, g_lsigma)

        sf = stan(model_code=code, data=dict(y=y), iter=200)
        mu = 0.1
        sigma = 2
        self.assertEqual(sf.log_prob(sf.unconstrain_pars(dict(mu=mu, sigma=sigma))),
                         log_prob_fun(mu, np.log(sigma)))
        self.assertEqual(sf.log_prob(sf.unconstrain_pars(dict(mu=mu, sigma=sigma)), False),
                         log_prob_fun(mu, np.log(sigma), adjust=False))
        g1 = sf.grad_log_prob(sf.unconstrain_pars(dict(mu=mu, sigma=sigma)), False)
        np.testing.assert_allclose(g1, log_prob_grad_fun(mu, np.log(sigma), adjust=False))

    def test_specify_args(self):
        y = (0.70,  -0.16,  0.77, -1.37, -1.99,  1.35, 0.08,
                0.02,  -1.48, -0.08,  0.34,  0.03, -0.42, 0.87,
                -1.36,  1.43,  0.80, -0.48, -1.61, -1.27)
        code = """
            data {
                real y[20];
            }
            parameters {
                real mu;
                real<lower=0> sigma;
            }
            model {
                y ~ normal(mu, sigma);
            }"""
        stepsize0 = 0.15
        sf = stan(model_code=code, data=dict(y=y), iter=200,
                    control=dict(adapt_engaged=False, stepsize=stepsize0))
        self.assertEqual(sf.get_sampler_params()[0]['stepsize__'][0], stepsize0)
        sf2 = stan(fit=sf, iter=20, algorithm='HMC', data=dict(y=y),
                    control=dict(adapt_engaged=False, stepsize=stepsize0))
        self.assertEqual(sf2.get_sampler_params()[0]['stepsize__'][0], stepsize0)
        sf3 = stan(fit=sf, iter=1, data=dict(y=y), init=0, chains=1)
        i_u = sf3.unconstrain_pars(sf3.get_inits()[0])
        np.testing.assert_equal(i_u, [0, 0])
