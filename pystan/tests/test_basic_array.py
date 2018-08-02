import unittest

import numpy as np

import pystan


class TestBasicArray(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = """
        data {
          int<lower=2> K;
        }
        parameters {
          real beta[K,1,2];
        }
        model {
          for (k in 1:K)
            beta[k,1,1] ~ normal(0,1);
          for (k in 1:K)
            beta[k,1,2] ~ normal(100,1);
        }"""
        cls.model = pystan.StanModel(model_code=model_code)
        cls.fit = cls.model.sampling(data=dict(K=4))

    def test_array_param_sampling(self):
        """
        Make sure shapes are getting unraveled correctly. Mixing up row-major and
        column-major data is a potential issue.
        """
        fit = self.fit

        # extract, permuted
        beta = fit.extract()['beta']
        self.assertEqual(beta.shape, (4000, 4, 1, 2))
        beta_mean = np.mean(beta, axis=0)
        self.assertEqual(beta_mean.shape, (4, 1, 2))
        self.assertTrue(np.all(beta_mean[:, 0, 0] < 4))
        self.assertTrue(np.all(beta_mean[:, 0, 1] > 100 - 4))

        # extract, permuted=False
        extracted = fit.extract(permuted=False)
        self.assertEqual(extracted.shape, (1000, 4, 9))
        # in theory 0:4 should be
        # 'beta[0,0,0]'
        # 'beta[1,0,0]'
        # 'beta[2,0,0]'
        # 'beta[3,0,0]'
        #
        # and 4:8 should be
        # 'beta[0,0,1]'
        # 'beta[1,0,1]'
        # 'beta[2,0,1]'
        # 'beta[3,0,1]'
        self.assertTrue(np.all(np.mean(extracted[:, :, 0:4], axis=(0, 1)) < 4))
        self.assertTrue(np.all(np.mean(extracted[:, :, 4:8], axis=(0, 1)) > 100 - 4))
        self.assertTrue(np.all(extracted[:, :, 8] < 0))  # lp__

    def test_array_param_optimizing(self):
        fit = self.fit
        sm = fit.stanmodel
        op = sm.optimizing(data=dict(K=4))
        beta = op['beta']
        self.assertEqual(beta.shape, (4, 1, 2))
        self.assertTrue(np.all(beta[:, 0, 0] < 4))
        self.assertTrue(np.all(beta[:, 0, 1] > 100 - 4))
