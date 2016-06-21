import unittest

import pystan
import pystan.chains
import pystan._chains


class TestESS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        model = pystan.StanModel(model_code=model_code, model_name="normal2")
        cls.fit = model.sampling(iter=4000, chains=2, seed=5)

    def test_ess(self):
        sim = self.fit.sim
        ess = pystan.chains.ess(sim, 0)
        self.assertAlmostEqual(ess, 1333.7, delta=1)
        ess2 = pystan.chains.ess(sim, 1)
        self.assertAlmostEqual(ess2, 1062.4, delta=3)

    def test_autocovariance(self):
        sim = self.fit.sim
        chain = 0
        param = 0
        acov = pystan._chains._test_autocovariance(sim, chain, param)
        self.assertAlmostEqual(sum(acov), -0.36232950982168854)
        acov = pystan._chains._test_autocovariance(sim, chain, param + 1)
        self.assertAlmostEqual(sum(acov), -1.9305054433755608)

    def test_stan_functions(self):
        pystan._chains._test_stan_functions()
