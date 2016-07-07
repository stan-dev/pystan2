import unittest


import pystan


class TestNormalVB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        cls.model = pystan.StanModel(model_code=model_code, model_name="normal1",
                                     verbose=True, obfuscate_model_name=False)

    def test_constructor(self):
        self.assertEqual(self.model.model_name, "normal1")
        self.assertEqual(self.model.model_cppname, "normal1")

    def test_log_prob(self):
        fit = self.model.sampling()
        extr = fit.extract()
        y_last, log_prob_last = extr['y'][-1], extr['lp__'][-1]
        self.assertEqual(fit.log_prob(y_last), log_prob_last)

    def test_vb_default(self):
        vbf = self.model.vb()
        self.assertIsNotNone(vbf)

    def test_vb_fullrank(self):
        vbf = self.model.vb(algorithm='fullrank')
        self.assertIsNotNone(vbf)
