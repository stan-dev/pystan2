import unittest

import pystan
from pystan._compat import PY2


class TestUnconstrainPars(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = pystan.StanModel(model_code="parameters { real x; } model { }")

    def test_unconstrain_pars(self):
        data, seed = {}, 1
        fit = self.model.fit_class(data, seed)
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(RuntimeError, 'Variable x missing'):
            fit.unconstrain_pars({})
