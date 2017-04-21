import unittest

import numpy as np

import pystan
from pystan._compat import PY2


class TestArgs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        cls.model = pystan.StanModel(model_code=model_code)

    def test_control(self):
        model = self.model
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(ValueError, '`control` must be a dictionary'):
            control_invalid = 3
            model.sampling(control=control_invalid)
        with assertRaisesRegex(ValueError, '`control` contains unknown'):
            control_invalid = dict(foo=3)
            model.sampling(control=control_invalid)
        with assertRaisesRegex(ValueError, '`metric` must be one of'):
            model.sampling(control={'metric': 'lorem-ipsum'})
