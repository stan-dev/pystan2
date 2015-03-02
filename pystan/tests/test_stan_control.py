import unittest

import pystan


class TestStanControl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        cls.model = pystan.StanModel(model_code=model_code)

    def test_stepsize(self):
        fit = self.model.sampling(control=dict(stepsize=0.001))
        self.assertIsNotNone(fit)
