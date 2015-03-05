from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import io
import os
import tempfile

import numpy as np

from pystan import stan, stanc

class TestStanFileIO(unittest.TestCase):

    def test_stan_model_from_file(self):
        bernoulli_model_code = """
            data {
            int<lower=0> N;
            int<lower=0,upper=1> y[N];
            }
            parameters {
            real<lower=0,upper=1> theta;
            }
            model {
            for (n in 1:N)
                y[n] ~ bernoulli(theta);
            }
            """

        bernoulli_data = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}

        temp_dir = tempfile.mkdtemp()
        temp_fn = os.path.join(temp_dir, 'modelcode.stan')
        with io.open(temp_fn, 'wt') as outfile:
            outfile.write(bernoulli_model_code)

        code = stanc(file=temp_fn)['model_code']
        fit = stan(model_code=code, data=bernoulli_data)
        extr = fit.extract(permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02

        # permuted=False
        extr = fit.extract(permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4

        # permuted=True
        extr = fit.extract('lp__', permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        extr = fit.extract('theta', permuted=True)
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02
        extr = fit.extract('theta', permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4

        fit = stan(file=temp_fn, data=bernoulli_data)
        extr = fit.extract(permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02

        # permuted=False
        extr = fit.extract(permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4

        # permuted=True
        extr = fit.extract('lp__', permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        extr = fit.extract('theta', permuted=True)
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02
        extr = fit.extract('theta', permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4

        os.remove(temp_fn)
