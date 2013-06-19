import time
import unittest

import numpy as np

from pystan import stan


class TestExtract(unittest.TestCase):

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

    fit = stan(model_code=ex_model_code, chains=4)

    def test_extract_permuted(self):
        ss = self.fit.extract(permuted=True)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        assert ss.keys() == {'alpha', 'beta', 'lp__'}
        print(alpha.shape)
        print(beta.shape)
        print(lp__.shape)
        assert alpha.shape == (4000, 2, 3)
        assert beta.shape == (4000, 2)
        assert lp__.shape == (4000,)

        # extract one at a time
        alpha2 = self.fit.extract('alpha', permuted=True)['alpha']
        assert alpha2.shape == (4000, 2, 3)
        np.testing.assert_array_equal(alpha, alpha2)
        beta = self.fit.extract('beta', permuted=True)['beta']
        assert beta.shape == (4000, 2)
        lp__ = self.fit.extract('lp__', permuted=True)['lp__']
        assert lp__.shape == (4000,)

    def test_extract_permuted_false(self):
        ss = self.fit.extract(permuted=False)
        assert ss.shape == (1000, 4, 9)
