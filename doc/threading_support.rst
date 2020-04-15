.. _threading_support:

.. currentmodule:: pystan

===================================
Threading Support with Pystan 2.18+
===================================

Notice! This is an experimental feature and is not tested or supported officially with PyStan 2.
Official multithreading support will land with PyStan 3.

By default, ``stan-math`` is not thread safe. Stan 2.18+ has ability to switch on
threading support with compile time arguments.

See https://github.com/stan-dev/math/wiki/Threading-Support

Due to use of ``multiprocessing`` to parallelize chains, user needs to be aware of the cpu usage.
This means that each chain will use ``STAN_NUM_THREADS`` cpu cores and this can have an affect on performance.

Windows
=======

These instructions are invalid on Windows with MingW-W64 compiler and should not be used.
Usage will crash the current Python session, which means that no sampling can be done.

see https://github.com/Alexpux/MINGW-packages/issues/2519 and https://sourceforge.net/p/mingw-w64/bugs/445/

Example
=======

.. code-block:: python

    import pystan
    import os
    import sys

    # set environmental variable STAN_NUM_THREADS
    # Use 4 cores per chain
    os.environ['STAN_NUM_THREADS'] = "4"

    # Example model
    # see http://discourse.mc-stan.org/t/cant-make-cmdstan-2-18-in-windows/5088/18
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

    extra_compile_args = ['-pthread', '-DSTAN_THREADS']

    stan_model = pystan.StanModel(
        model_code=stan_code,
        extra_compile_args=extra_compile_args
    )

    # use the default 4 chains == 4 parallel process
    # used cores = min(cpu_cores, 4*STAN_NUM_THREADS)
    fit = stan_model.sampling(data=stan_data, n_jobs=4)

    print(fit)
