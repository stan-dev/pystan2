import unittest

import pystan


class Test8Schools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        schools_code = """\
            data {
                int<lower=0> J; // number of schools
                real y[J]; // estimated treatment effects
                real<lower=0> sigma[J]; // s.e. of effect estimates
            }
            parameters {
                real mu;
                real<lower=0> tau;
                real eta[J];
            }
            transformed parameters {
                real theta[J];
                for (j in 1:J)
                    theta[j] <- mu + tau * eta[j];
            }
            model {
                eta ~ normal(0, 1);
                y ~ normal(theta, sigma);
            }"""

        cls.schools_dat = {'J': 8,
                           'y': [28,  8, -3,  7, -1,  1, 18, 12],
                           'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

        cls.model = pystan.StanModel(model_code=schools_code)

    def test_8schools_pars(self):
        model = self.model
        data = self.schools_dat
        pars = ['mu']  # list of parameters
        fit = model.sampling(data=data, pars=pars)
        self.assertEqual(len(fit.sim['pars_oi']), len(fit.sim['dims_oi']))
        self.assertEqual(len(fit.extract()), 2)
        self.assertIn('mu', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))

        pars = ['eta']
        fit = model.sampling(data=data, pars=pars)
        self.assertEqual(len(fit.extract()), 2)
        self.assertIn('eta', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))

        pars = ['mu', 'eta']
        fit = model.sampling(data=data, pars=pars)
        self.assertEqual(len(fit.extract()), 3)
        self.assertIn('mu', fit.extract())
        self.assertIn('eta', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))

    def test_8schools_pars_bare(self):
        model = self.model
        data = self.schools_dat
        pars = 'mu'  # bare string
        fit = model.sampling(data=data, pars=pars)
        self.assertEqual(len(fit.extract()), 2)
        self.assertIn('mu', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))

        pars = 'eta'  # bare string
        fit = model.sampling(data=data, pars=pars)
        self.assertEqual(len(fit.extract()), 2)
        self.assertIn('eta', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))
