import unittest

import numpy as np

from pystan import stan

# REF: rstan/tests/unitTests/runit.test.stan_args_hpp.R


class TestPyStanArgs(unittest.TestCase):

    def test_pystanags(self):
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

        sf = stan(model_code=code, iter=100, thin=3, data=dict(y=y))
        args = sf.stan_args[0]
        self.assertEqual(args['iter'], 100)
        self.assertEqual(args['thin'], 3)
        self.assertEqual(args['init'], b'random')

        sampling = args['ctrl']['sampling']
        self.assertEqual(sampling['adapt_engaged'], True)
        self.assertEqual(sampling['adapt_window'], 25)
        self.assertEqual(sampling['adapt_init_buffer'], 75)
        self.assertEqual(sampling['adapt_gamma'], 0.05)
        self.assertEqual(sampling['adapt_delta'], 0.8)
        self.assertEqual(sampling['adapt_kappa'], 0.75)
        self.assertEqual(sampling['adapt_t0'], 10)
