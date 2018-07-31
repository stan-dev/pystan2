import unittest

import numpy as np

import pystan
from pystan._compat import PY2
from pystan.tests.helper import get_model


class TestArgs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real x;real y;real z;} model {x ~ normal(0,1);y ~ normal(0,1);z ~ normal(0,1);}'
        cls.model = get_model("standard_normals_model", model_code)
        #cls.model = pystan.StanModel(model_code=model_code)

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

    def test_print_summary(self):
        model = self.model
        fit = model.sampling(iter=100)

        summary_full = pystan.misc.stansummary(fit)
        summary_one_par1 = pystan.misc.stansummary(fit, pars='z')
        summary_one_par2 = pystan.misc.stansummary(fit, pars=['z'])
        summary_pars = pystan.misc.stansummary(fit, pars=['x', 'y'])

        self.assertNotEqual(summary_full, summary_one_par1)
        self.assertNotEqual(summary_full, summary_one_par2)
        self.assertNotEqual(summary_full, summary_pars)
        self.assertNotEqual(summary_one_par1, summary_pars)
        self.assertNotEqual(summary_one_par2, summary_pars)

        self.assertEqual(summary_one_par1, summary_one_par2)
