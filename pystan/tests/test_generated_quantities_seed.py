from collections import OrderedDict
import gc
import os
import tempfile
import unittest

import numpy as np

import pystan
from pystan._compat import PY2


class TestGeneratedQuantitiesSeed(unittest.TestCase):
    """Verify that the RNG in the transformed data block uses the overall seed.

    See https://github.com/stan-dev/stan/issues/2241

    """

    @classmethod
    def setUpClass(cls):
        model_code = """
            data {
              int<lower=0> N;
            }
            transformed data {
              vector[N] y;
              for (n in 1:N)
                y[n] = normal_rng(0, 1);
            }
            parameters {
              real mu;
              real<lower = 0> sigma;
            }
            model {
              y ~ normal(mu, sigma);
            }
            generated quantities {
              real mean_y = mean(y);
              real sd_y = sd(y);
            }
        """
        cls.model = pystan.StanModel(model_code=model_code, verbose=True)

    def test_generated_quantities_seed(self):
        fit1 = self.model.sampling(data={'N': 1000}, iter=10, seed=123)
        extr1 = fit1.extract()
        fit2 = self.model.sampling(data={'N': 1000}, iter=10, seed=123)
        extr2 = fit2.extract()
        self.assertTrue((extr1['mean_y'] == extr2['mean_y']).all())
        fit3 = self.model.sampling(data={'N': 1000}, iter=10, seed=456)
        extr3 = fit3.extract()
        self.assertFalse((extr1['mean_y'] == extr3['mean_y']).all())
