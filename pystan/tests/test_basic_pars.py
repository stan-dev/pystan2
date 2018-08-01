import unittest
from pystan.tests.helper import get_model


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
                    theta[j] = mu + tau * eta[j];
            }
            model {
                eta ~ normal(0, 1);
                y ~ normal(theta, sigma);
            }"""

        cls.schools_dat = {'J': 8,
                           'y': [28,  8, -3,  7, -1,  1, 18, 12],
                           'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

        cls.model = get_model("schools_model", schools_code)
        #cls.model pystan.StanModel(model_code=schools_code)

    def test_8schools_pars(self):
        model = self.model
        data = self.schools_dat
        pars = ['mu']  # list of parameters
        fit = model.sampling(data=data, pars=pars, iter=100)
        self.assertEqual(len(fit.sim['pars_oi']), len(fit.sim['dims_oi']))
        self.assertEqual(len(fit.extract()), 2)
        self.assertIn('mu', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))

        pars = ['eta']
        fit = model.sampling(data=data, pars=pars, iter=100)
        self.assertEqual(len(fit.extract()), 2)
        self.assertIn('eta', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))

        pars = ['mu', 'eta']
        fit = model.sampling(data=data, pars=pars, iter=100)
        self.assertEqual(len(fit.extract()), 3)
        self.assertIn('mu', fit.extract())
        self.assertIn('eta', fit.extract())
        self.assertRaises(ValueError, fit.extract, 'theta')
        self.assertIsNotNone(fit.extract(permuted=False))

    def test_8schools_pars_bare(self):
        model = self.model
        data = self.schools_dat
        pars = 'mu'  # bare string
        fit = model.sampling(data=data, pars=pars, iter=100)
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

    def test_8schools_bad_pars(self):
        model = self.model
        data = self.schools_dat
        pars = ['mu', 'missing']  # 'missing' is not in the model
        self.assertRaises(ValueError, model.sampling,
                          data=data, pars=pars, iter=100)


class TestParsLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = '''\
            parameters {
              real a;
              real b;
              real c;
              real d;
            }
            model {
              a ~ normal(0, 1);
              b ~ normal(-19, 1);
              c ~ normal(5, 1);
              d ~ normal(11, 1);
            }
        '''
        cls.model = pystan.StanModel(model_code=model_code)

    def test_pars_single(self):
        '''Make sure labels are not getting switched'''
        fit = self.model.sampling(iter=100, chains=1, n_jobs=1, pars=['d'])
        self.assertGreater(fit.extract()['d'].mean(), 7)

    def test_pars_single_chains(self):
        '''Make sure labels are not getting switched'''
        fit = self.model.sampling(iter=100, chains=2, n_jobs=2, pars=['d'])
        self.assertGreater(fit.extract()['d'].mean(), 7)

    def test_pars_labels(self):
        '''Make sure labels are not getting switched'''
        fit = self.model.sampling(iter=100, chains=2, n_jobs=2, pars=['a', 'd'])
        extr = fit.extract()
        a = extr['a'].mean()
        d = extr['d'].mean()
        self.assertTrue('b' not in extr)
        self.assertTrue('c' not in extr)
        self.assertGreater(d, a)
