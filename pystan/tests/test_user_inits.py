import unittest

import pystan
from pystan._compat import PY2

class TestUserInits(unittest.TestCase):

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

    data = dict(x=2)

    def test_user_init(self):
        model_code = self.model_code
        data = self.data
        fit1 = pystan.stan(model_code=model_code, iter=10, chains=1, seed=2,
                           data=data, init=[dict(mu=4)], warmup=0)
        self.assertEqual(fit1.get_inits()[0]['mu'], 4)
        fit2 = pystan.stan(model_code=model_code, iter=10, chains=1, seed=2,
                           data=data, init=[dict(mu=400)], warmup=0)
        self.assertEqual(fit2.get_inits()[0]['mu'], 400)
        self.assertFalse(all(fit1.extract()['mu'] == fit2.extract()['mu']))

    def test_user_initfun(self):
        model_code = self.model_code
        data = self.data

        def make_inits(chain_id):
            return dict(mu=chain_id)

        fit1 = pystan.stan(model_code=model_code, iter=10, chains=4, seed=2,
                           data=data, init=make_inits, warmup=0)
        for i, inits in enumerate(fit1.get_inits()):
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
        # NOTE: we are only specifying 'mu' and not 'sigma'
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(RuntimeError, "sigma missing"):
            pystan.stan(model_code=model_code, iter=10, chains=1, seed=2,
                        data=data, init=[dict(mu=4)], warmup=0)
