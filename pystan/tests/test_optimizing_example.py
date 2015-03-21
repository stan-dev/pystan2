import unittest

import numpy as np

import pystan


class TestOptimizingExample(unittest.TestCase):
    """Test optimizing example from documentation"""

    @classmethod
    def setUpClass(cls):
        ocode = """
        data {
            int<lower=1> N;
            real y[N];
        }
        parameters {
            real mu;
        }
        model {
            y ~ normal(mu, 1);
        }
        """

        cls.sm = pystan.StanModel(model_code=ocode)

    def test_optimizing(self):
        sm = self.sm
        np.random.seed(3)
        y2 = np.random.normal(size=20)
        op = sm.optimizing(data=dict(y=y2, N=len(y2)))
        self.assertAlmostEqual(op['mu'], np.mean(y2))
