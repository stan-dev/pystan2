import unittest

import pystan
import pystan.chains
import pystan._chains

# NOTE: This test is fragile because there is no guarantee that samples are
# consistent between Stan releases (even with fixed random seeds).  Consider
# this a smoke test.

class TestESS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        model = pystan.StanModel(model_code=model_code, model_name="normal2")
        cls.fit = model.sampling(iter=4000, chains=2, seed=5)

    def test_ess(self):
        sim = self.fit.sim
        ess = pystan.chains.ess(sim, sim['fnames_oi'].index('y'))
        self.assertGreater(ess, 2000)
