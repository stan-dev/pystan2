import unittest

import pystan
import pystan.chains
import pystan._chains

# NOTE: This test is fragile because the default sampling algorithm used by Stan
# may change in significant ways.


class TestESS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        model = pystan.StanModel(model_code=model_code)
        cls.fits = [model.sampling(iter=4000, chains=2, seed=i) for i in range(10)]

    def test_ess(self):
        sims = [fit.sim for fit in self.fits]
        esses = [pystan.chains.ess(sim, sim['fnames_oi'].index('y')) for sim in sims]
        # in Stan 2.15 the default acceptance rate was changed, ess dropped
        self.assertGreater(sum(esses) / len(esses), 1200)
