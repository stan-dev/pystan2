import unittest

from pystan import StanModel

class TestCovExpQuad(unittest.TestCase):
    """
    Unit test added to address issue discussed here: https://github.com/stan-dev/pystan/issues/271
    """
    @classmethod
    def setUpClass(cls):
      covexpquad = """
          data {
          real rx1[5];
      }
      model {
          matrix[5,5] a;

          a = cov_exp_quad(rx1, 1, 1);
      }
          """
      cls.model = StanModel(model_code=covexpquad)

    def test_covexpquad(self):
        model = self.model
        self.assertIsNotNone(model)
