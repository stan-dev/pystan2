import unittest

import numpy as np

import pystan
from pystan.tests.helper import get_model

# REF: rstan/tests/unitTests/runit.test.stan_args_hpp.R


class TestPyStanArgs(unittest.TestCase):

    def test_stan_args_basic(self):
        y = np.array([0.70,  -0.16,  0.77, -1.37, -1.99,  1.35, 0.08,
                      0.02,  -1.48, -0.08,  0.34,  0.03, -0.42, 0.87,
                      -1.36,  1.43,  0.80, -0.48, -1.61, -1.27])

        code = '''
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
        }'''
        sm = get_model("normal_mu_sigma_model", code)
        sf = sm.sampling(iter=10, thin=3, data=dict(y=y, N=20))
        args = sf.stan_args[0]
        self.assertEqual(args['iter'], 10)
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

    def test_stan_args_optimizing(self):
        args = pystan.misc._get_valid_stan_args(dict(method="optim"))
        # default optimizing algorithm is LBFGS
        self.assertEqual(args['ctrl']['optim']['algorithm'], pystan.constants.optim_algo_t.LBFGS)

        args = pystan.misc._get_valid_stan_args(dict(iter=100, seed=12345, method='optim'))
        self.assertEqual(args['random_seed'], 12345)
        self.assertEqual(args['ctrl']['optim']['algorithm'], pystan.constants.optim_algo_t.LBFGS)

        args = pystan.misc._get_valid_stan_args(dict(iter=100, seed=12345, method='optim', algorithm='BFGS'))
        self.assertEqual(args['random_seed'], 12345)
        self.assertEqual(args['ctrl']['optim']['algorithm'], pystan.constants.optim_algo_t.BFGS)

        args = pystan.misc._get_valid_stan_args(dict(iter=100, seed=12345, method='optim', algorithm='LBFGS'))
        self.assertEqual(args['random_seed'], 12345)
        self.assertEqual(args['ctrl']['optim']['algorithm'], pystan.constants.optim_algo_t.LBFGS)
        self.assertEqual(args['ctrl']['optim']['history_size'], 5)

        args = pystan.misc._get_valid_stan_args(dict(iter=100, method='optim', history_size=6))
        self.assertEqual(args['ctrl']['optim']['history_size'], 6)

    def test_stan_args_sampling(self):
        # defaults are largely controlled by
        # pystan.misc_get_valid_stan_args
        # which has an analog in rstan, rstan::stan_args
        args = pystan.misc._get_valid_stan_args(dict(iter=100, thin=100))
        self.assertEqual(args['ctrl']['sampling']['iter'], 100)
        self.assertEqual(args['ctrl']['sampling']['thin'], 100)

        args = pystan.misc._get_valid_stan_args(dict(iter=5, thin=3, refresh=-1, seed="12345"))
        self.assertEqual(args['ctrl']['sampling']['iter'], 5)
        self.assertEqual(args['ctrl']['sampling']['thin'], 3)
        self.assertEqual(args['ctrl']['sampling']['refresh'], -1)
        self.assertEqual(args['random_seed'], '12345')

        args = pystan.misc._get_valid_stan_args(dict(iter=5, thin=3, refresh=-1, seed="12345", method='test_grad'))
        self.assertEqual(args['method'], pystan.constants.stan_args_method_t.TEST_GRADIENT)

        args = pystan.misc._get_valid_stan_args(dict(method='test_grad', epsilon=1.3, error=0.1))
        self.assertEqual(args['method'], pystan.constants.stan_args_method_t.TEST_GRADIENT)
        self.assertEqual(args['ctrl']['test_grad']['epsilon'], 1.3)
        self.assertEqual(args['ctrl']['test_grad']['error'], 0.1)

        args = pystan.misc._get_valid_stan_args(dict(iter=100, algorithm='HMC'))
        self.assertEqual(args['ctrl']['sampling']['algorithm'], pystan.constants.sampling_algo_t.HMC)

        args = pystan.misc._get_valid_stan_args(dict(algorithm='NUTS'))
        self.assertEqual(args['ctrl']['sampling']['algorithm'], pystan.constants.sampling_algo_t.NUTS)

        args = pystan.misc._get_valid_stan_args(dict(algorithm='NUTS', control=dict(stepsize=0.1)))
        self.assertEqual(args['ctrl']['sampling']['algorithm'], pystan.constants.sampling_algo_t.NUTS)
        self.assertEqual(args['ctrl']['sampling']['stepsize'], 0.1)

        args = pystan.misc._get_valid_stan_args(dict(algorithm='NUTS', control=dict(stepsize=0.1, metric='unit_e')))
        self.assertEqual(args['ctrl']['sampling']['algorithm'], pystan.constants.sampling_algo_t.NUTS)
        self.assertEqual(args['ctrl']['sampling']['stepsize'], 0.1)
        self.assertEqual(args['ctrl']['sampling']['metric'], pystan.constants.sampling_metric_t.UNIT_E)

        args = pystan.misc._get_valid_stan_args(dict(algorithm='NUTS', control=dict(stepsize=0.1, metric='diag_e')))
        self.assertEqual(args['ctrl']['sampling']['algorithm'], pystan.constants.sampling_algo_t.NUTS)
        self.assertEqual(args['ctrl']['sampling']['stepsize'], 0.1)
        self.assertEqual(args['ctrl']['sampling']['metric'], pystan.constants.sampling_metric_t.DIAG_E)

        args = pystan.misc._get_valid_stan_args(dict(algorithm='NUTS', control=dict(stepsize=0.1, metric='diag_e',
                                                adapt_term_buffer=4, adapt_window=30, adapt_init_buffer=40)))
        self.assertEqual(args['ctrl']['sampling']['algorithm'], pystan.constants.sampling_algo_t.NUTS)
        self.assertEqual(args['ctrl']['sampling']['stepsize'], 0.1)
        self.assertEqual(args['ctrl']['sampling']['metric'], pystan.constants.sampling_metric_t.DIAG_E)
        self.assertEqual(args['ctrl']['sampling']['adapt_term_buffer'], 4)
        self.assertEqual(args['ctrl']['sampling']['adapt_window'], 30)
        self.assertEqual(args['ctrl']['sampling']['adapt_init_buffer'], 40)
