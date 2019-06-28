import unittest

import numpy as np

import pystan


class TestUserInits(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = """
            data {
              real x;
            }
            parameters {
              real mu;
            }
            model {
              x ~ normal(mu,1);
            }
        """
        cls.data = dict(x=2)
        cls.model = pystan.StanModel(model_code=model_code)

    def test_user_init(self):
        model, data = self.model, self.data
        fit1 = model.sampling(iter=10, chains=1, seed=2, data=data, init=[dict(mu=4)], warmup=0, control={"adapt_engaged" : False})
        self.assertEqual(fit1.get_inits()[0]['mu'], 4)
        fit2 = model.sampling(iter=10, chains=1, seed=2, data=data, init=[dict(mu=400)], warmup=0, control={"adapt_engaged" : False})
        self.assertEqual(fit2.get_inits()[0]['mu'], 400)
        self.assertFalse(all(fit1.extract()['mu'] == fit2.extract()['mu']))

    def test_user_initfun(self):
        model, data = self.model, self.data

        def make_inits(chain_id):
            return dict(mu=chain_id)

        fit1 = model.sampling(iter=10, chains=4, seed=2, data=data, init=make_inits, warmup=0, control={"adapt_engaged" : False})
        for i, inits in enumerate(fit1.get_inits()):
            self.assertEqual(inits['mu'], i)

    def test_user_initfun_chainid(self):
        model, data = self.model, self.data

        def make_inits(chain_id):
            return dict(mu=chain_id)

        chain_id = [9, 10, 11, 12]
        fit1 = model.sampling(iter=10, chains=4, seed=2, data=data,
                              init=make_inits, warmup=0, chain_id=chain_id, control={"adapt_engaged" : False})
        for i, inits in zip(chain_id, fit1.get_inits()):
            self.assertEqual(inits['mu'], i)

    def test_user_init_unspecified(self):
        model_code = """
        data {
          real x;
        }
        parameters {
          real mu;
          real<lower=0> sigma;
        }
        model {
          x ~ normal(mu, sigma);
        }
        """
        data = self.data
        # NOTE: we are only specifying 'mu' and not 'sigma' (partial inits)
        fit = pystan.stan(model_code=model_code, iter=10, chains=1, seed=2, data=data, init=[dict(mu=4)], warmup=0, control={"adapt_engaged" : False})
        self.assertIsNotNone(fit)


class TestUserInitsMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = """
            data {
              int<lower=2> K;
              int<lower=1> D;
            }
            parameters {
              matrix[K,D] beta;
            }
            model {
            for (k in 1:K)
                for (d in 1:D)
                  beta[k,d] ~ normal((d==2?100:0),1);
           }
        """
        cls.model = pystan.StanModel(model_code=model_code)
        cls.data = dict(K=3, D=4)

    def test_user_init(self):
        model, data = self.model, self.data
        beta = np.ones((data['K'], data['D']))
        fit1 = model.sampling(iter=10, chains=1, seed=2,
                              data=data, init=[dict(beta=beta)], warmup=0, control={"adapt_engaged" : False})
        np.testing.assert_equal(fit1.get_inits()[0]['beta'], beta)
        beta = 5 * np.ones((data['K'], data['D']))
        fit2 = model.sampling(iter=10, chains=1, seed=2,
                              data=data, init=[dict(beta=beta)], warmup=0, control={"adapt_engaged" : False})
        np.testing.assert_equal(fit2.get_inits()[0]['beta'], beta)

    def test_user_initfun(self):
        model, data = self.model, self.data

        beta = np.ones((data['K'], data['D']))

        def make_inits(chain_id):
            return dict(beta=beta * chain_id)

        fit1 = model.sampling(iter=10, chains=4, seed=2, data=data, init=make_inits, warmup=0, control={"adapt_engaged" : False})
        for i, inits in enumerate(fit1.get_inits()):
            np.testing.assert_equal(beta * i, inits['beta'])
