import unittest

import numpy as np

import pystan
from pystan._compat import PY2


class TestFixedParam(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = '''
            model { }
            generated quantities {
              real y;
              y = normal_rng(0, 1);
            }
        '''
        cls.model = pystan.StanModel(model_code=model_code)

    def test_fixed_param(self):
        fit = self.model.sampling(algorithm='Fixed_param')
        self.assertEqual(fit.sim['iter'], 2000)
        self.assertEqual(fit.sim['pars_oi'], ['y', 'lp__'])
        self.assertEqual(len(fit.sim['samples']), 4)
        for sample_dict in fit.sim['samples']:
            assert -0.5 < np.mean(sample_dict['chains']['y']) < 0.5
            assert 0.5 < np.std(sample_dict['chains']['y']) < 1.5
