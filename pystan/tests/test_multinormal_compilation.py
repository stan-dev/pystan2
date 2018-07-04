import unittest

import numpy as np
from pystan import StanModel


class TestMultiNormalCompilation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
      
        model_code = """
        data {
          int<lower=0> M;
          int<lower=0> N;
          vector[N] y;
          matrix[N, M] A;
          matrix[N, N] Sigma;
        }
        parameters {
          vector[M] x;
        }
        model {
          y ~ multi_normal(A * x, Sigma);
        }"""
        cls.model = StanModel(model_code=model_code)

    def test_compilation(self):
        pass
