import unittest

import numpy as np

import pystan
from pystan._compat import PY2


class TestArgs(unittest.TestCase):

    def test_control(self):
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'

        with assertRaisesRegex(ValueError, '`control` must be a dictionary'):
            control_invalid = 3
            pystan.stan(model_code=model_code, control=control_invalid)
        with assertRaisesRegex(ValueError, '`control` contains unknown'):
            control_invalid = dict(foo=3)
            pystan.stan(model_code=model_code, control=control_invalid)
        with assertRaisesRegex(ValueError, '`metric` must be one of'):
            pystan.stan(model_code=model_code, control={'metric': 'lorem-ipsum'})
