import unittest

import numpy as np
from pystan import StanModel


class TestMatrixParam(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = """
        data {
        int<lower=2> K;
        int<lower=1> D;
        }
        parameters {
        matrix[K,D] beta;
        }
        model {
        for (k in 1:K)
            for (d in 1:D)
            beta[k,d] ~ normal(if_else(d==2,100, 0),1);
        }"""
        cls.model = StanModel(model_code=model_code)

    def test_matrix_param(self):
        sm = self.model
        fit = sm.sampling(data=dict(K=3, D=4))
        beta = fit.extract()['beta']
        assert beta.shape == (4000, 3, 4)
        assert np.mean(beta[:, 0, 0]) < 4
        extracted = fit.extract(permuted=False)
        assert extracted.shape == (1000, 4, 13)
        assert np.mean(extracted[:,:,0]) < 4
        assert np.all(extracted[:,:,12] < 0)  # lp__

    def test_matrix_param_order(self):
        sm = self.model
        fit = sm.sampling(data=dict(K=3, D=2))
        beta = fit.extract()['beta']
        assert beta.shape == (4000, 3, 2)
        beta_mean = np.mean(beta, axis=0)
        beta_colmeans = np.mean(beta_mean, axis=0)
        assert beta_colmeans[0] < 4
        assert beta_colmeans[1] > 100 - 4

    def test_matrix_param_order_optimizing(self):
        sm = self.model
        op = sm.optimizing(data=dict(K=3, D=2))
        beta = op['beta']
        assert beta.shape == (3, 2)
        beta_colmeans = np.mean(beta, axis=0)
        assert beta_colmeans[0] < 4
        assert beta_colmeans[1] > 100 - 4
