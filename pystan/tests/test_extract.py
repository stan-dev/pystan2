import unittest

import numpy as np

import pystan


class TestExtract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ex_model_code = '''
        parameters {
            real alpha[2,3];
            real beta[2];
        }
        model {
            for (i in 1:2) for (j in 1:3)
            alpha[i, j] ~ normal(0, 1);
            for (i in 1:2)
            beta ~ normal(0, 2);
        }
        '''
        cls.sm = sm = pystan.StanModel(model_code=ex_model_code)
        cls.fit = sm.sampling(chains=4, iter=2000)

    def test_extract_permuted(self):
        ss = self.fit.extract(permuted=True)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        self.assertEqual(sorted(ss.keys()), sorted({'alpha', 'beta', 'lp__'}))
        self.assertEqual(alpha.shape, (4000, 2, 3))
        self.assertEqual(beta.shape, (4000, 2))
        self.assertEqual(lp__.shape, (4000,))
        self.assertTrue((~np.isnan(alpha)).all())
        self.assertTrue((~np.isnan(beta)).all())
        self.assertTrue((~np.isnan(lp__)).all())

        # extract one at a time
        alpha2 = self.fit.extract('alpha', permuted=True)['alpha']
        self.assertEqual(alpha2.shape, (4000, 2, 3))
        np.testing.assert_array_equal(alpha, alpha2)
        beta = self.fit.extract('beta', permuted=True)['beta']
        self.assertEqual(beta.shape, (4000, 2))
        lp__ = self.fit.extract('lp__', permuted=True)['lp__']
        self.assertEqual(lp__.shape, (4000,))

    def test_extract_permuted_false(self):
        fit = self.fit
        ss = fit.extract(permuted=False)
        num_samples = fit.sim['iter'] - fit.sim['warmup']
        self.assertEqual(ss.shape, (num_samples, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())

    def test_extract_permuted_false_inc_warmup(self):
        fit = self.fit
        ss = fit.extract(inc_warmup=True, permuted=False)
        num_samples = fit.sim['iter']
        self.assertEqual(ss.shape, (num_samples, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())

    def test_extract_thin(self):
        sm = self.sm
        fit = sm.sampling(chains=4, iter=2000, thin=2)

        # permuted True
        ss = fit.extract(permuted=True)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        self.assertEqual(sorted(ss.keys()), sorted({'alpha', 'beta', 'lp__'}))
        self.assertEqual(alpha.shape, (2000, 2, 3))
        self.assertEqual(beta.shape, (2000, 2))
        self.assertEqual(lp__.shape, (2000,))
        self.assertTrue((~np.isnan(alpha)).all())
        self.assertTrue((~np.isnan(beta)).all())
        self.assertTrue((~np.isnan(lp__)).all())

        # permuted False
        ss = fit.extract(permuted=False)
        self.assertEqual(ss.shape, (500, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())

        # permuted False inc_warmup True
        ss = fit.extract(inc_warmup=True, permuted=False)
        self.assertEqual(ss.shape, (1000, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())
