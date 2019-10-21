import distutils.errors
import sys
import os
import unittest

import pystan
from pystan._compat import PY2


class TestExtraCompileArgs(unittest.TestCase):

    def test_extra_compile_args(self):
        extra_compile_args = [
            '-O3',
            '-ftemplate-depth-1024',
            '-Wno-unused-function',
            '-Wno-uninitialized',
        ]
        if sys.platform.startswith("win"):
            extra_compile_args.extend([
                "-D_hypot=hypot",
                "-pthread",
                "-fexceptions"
            ])
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        model = pystan.StanModel(model_code=model_code, model_name="normal1",
                                 extra_compile_args=extra_compile_args)
        fit = model.sampling()
        extr = fit.extract()
        y_last, log_prob_last = extr['y'][-1], extr['lp__'][-1]
        self.assertEqual(fit.log_prob(y_last), log_prob_last)

    def test_extra_compile_args_failure(self):
        extra_compile_args = ['-non-existent-option']
        if sys.platform.startswith("win"):
            return
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(distutils.errors.CompileError, 'failed with exit status'):
            pystan.StanModel(model_code=model_code, model_name="normal1",
                             extra_compile_args=extra_compile_args)

    def test_threading_support(self):
        # Dont test with Windows with MinGW-w64 (GCC)
        if sys.platform.startswith("win"):
            return
        # Set up environmental variable
        os.environ['STAN_NUM_THREADS'] = "2"
        # Enable threading
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
        stan_code = """
        functions {
          vector bl_glm(vector mu_sigma, vector beta,
                        real[] x, int[] y) {
            vector[2] mu = mu_sigma[1:2];
            vector[2] sigma = mu_sigma[3:4];
            real lp = normal_lpdf(beta | mu, sigma);
            real ll = bernoulli_logit_lpmf(y | beta[1] + beta[2] * to_vector(x));
            return [lp + ll]';
          }
        }
        data {
          int<lower = 0> K;
          int<lower = 0> N;
          vector[N] x;
          int<lower = 0, upper = 1> y[N];
        }
        transformed data {
          int<lower = 0> J = N / K;
          real x_r[K, J];
          int<lower = 0, upper = 1> x_i[K, J];
          {
            int pos = 1;
            for (k in 1:K) {
              int end = pos + J - 1;
              x_r[k] = to_array_1d(x[pos:end]);
              x_i[k] = y[pos:end];
              pos += J;
            }
          }
        }
        parameters {
          vector[2] beta[K];
          vector[2] mu;
          vector<lower=0>[2] sigma;
        }
        model {
          mu ~ normal(0, 2);
          sigma ~ normal(0, 2);
          target += sum(map_rect(bl_glm, append_row(mu, sigma),
                                 beta, x_r, x_i));
        }
        """
        stan_data = dict(
            K = 4,
            N = 12,
            x = [1.204, -0.573, -1.35, -1.157,
                 -1.29, 0.515, 1.496, 0.918,
                 0.517, 1.092, -0.485, -2.157],
            y = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
        )
        stan_model = pystan.StanModel(
            model_code=stan_code,
            extra_compile_args=extra_compile_args
        )
        for i in range(10):
            try:
                fit = stan_model.sampling(data=stan_data, chains=2, iter=200, n_jobs=1)
                self.assertIsNotNone(fit)
                fit2 = stan_model.sampling(data=stan_data, chains=2, iter=200, n_jobs=2)
                self.assertIsNotNone(fit2)
                draw = fit.extract(pars=fit.model_pars+['lp__'], permuted=False)
                lp = {key : values[-1, 0] for key, values in draw.items() if key == 'lp__'}['lp__']
                draw = {key : values[-1, 0] for key, values in draw.items() if key != 'lp__'}
                draw = fit.unconstrain_pars(draw)
                self.assertEqual(fit.log_prob(draw), lp)
                draw2 = fit2.extract(pars=fit2.model_pars+['lp__'], permuted=False)
                lp2 = {key : values[-1, 0] for key, values in draw2.items() if key == 'lp__'}['lp__']
                draw2 = {key : values[-1, 0] for key, values in draw2.items() if key != 'lp__'}
                draw2 = fit2.unconstrain_pars(draw2)
                self.assertEqual(fit2.log_prob(draw2), lp2)
                break
            except AssertionError:
                if i < 9:
                    continue
                else:
                    raise
