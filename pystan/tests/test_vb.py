import os
import unittest
import tempfile

import pystan
from pystan.tests.helper import get_model

class TestNormalVB(unittest.TestCase):

    seed = 1

    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real x;real y;real z;} model {x ~ normal(0,1);y ~ normal(0,1);z ~ normal(0,1);}'
        cls.model = get_model("standard_normals_model", model_code)
        #cls.model = pystan.StanModel(model_code=model_code)

    def test_vb_default(self):
        vbf1 = self.model.vb(seed=self.seed)
        self.assertIsNotNone(vbf1)
        csv_fn = vbf1['args']['sample_file'].decode('ascii')
        with open(csv_fn) as f:
            csv_sample1 = f.readlines()[10]

        # vb run with same seed should yield identical results
        vbf2 = self.model.vb(seed=self.seed)
        self.assertIsNotNone(vbf2)
        csv_fn = vbf2['args']['sample_file'].decode('ascii')
        with open(csv_fn) as f:
            csv_sample2 = f.readlines()[10]
        self.assertEqual(csv_sample1, csv_sample2)

    def test_vb_fullrank(self):
        vbf = self.model.vb(algorithm='fullrank', seed=self.seed)
        self.assertIsNotNone(vbf)

    def test_vb_sample_file(self):
        sample_file = os.path.join(tempfile.mkdtemp(), 'vb-results.csv')
        vbf = self.model.vb(algorithm='fullrank', sample_file=sample_file, seed=self.seed)
        self.assertIsNotNone(vbf)
        self.assertEqual(vbf['args']['sample_file'].decode('utf-8'), sample_file)
        self.assertTrue(os.path.exists(sample_file))

    def test_vb_diagnostic_file(self):
        sample_file = os.path.join(tempfile.mkdtemp(), 'vb-results.csv')
        diag_file = os.path.join(tempfile.mkdtemp(), 'vb-diag.csv')
        vbf = self.model.vb(algorithm='fullrank', sample_file=sample_file, diagnostic_file=diag_file, seed=self.seed)
        self.assertIsNotNone(vbf)
        self.assertEqual(vbf['args']['diagnostic_file'].decode('utf-8'), diag_file)
        self.assertTrue(os.path.exists(diag_file))

    def test_vb_pars(self):
        vbfit = self.model.vb(algorithm='fullrank', pars=['y'], seed=self.seed)
        vbfit2 = self.model.vb(algorithm='fullrank', pars='z', seed=self.seed)
        self.assertIsNotNone(vbfit)
        self.assertIsNotNone(vbfit2)
        self.assertEqual(vbfit['mean_par_names'], ['y'])
        self.assertEqual(vbfit2['mean_par_names'], ['z'])
