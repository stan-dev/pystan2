import logging
import time
import unittest

import numpy as np

from pystan import StanModel


def test_model_constructor():
    m = StanModel(model_code='parameters {real y;} model {y ~ normal(0,1);}',
                  model_name="normal1")
    assert m.model_name == "normal1"


class TestBernoulli(unittest.TestCase):

    bernoulli_model_code = """
        data {
        int<lower=0> N;
        int<lower=0,upper=1> y[N];
        }
        parameters {
        real<lower=0,upper=1> theta;
        }
        model {
        for (n in 1:N)
            y[n] ~ bernoulli(theta);
        }
        """
    bernoulli_data = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}

    model = StanModel(model_code=bernoulli_model_code, model_name="bernoulli")

    def test_bernoulli_constructor(self):
        model = self.model
        assert model.model_name == "bernoulli"
        assert model.model_cppname.endswith("bernoulli")

    def test_bernoulli_compile_time(self):
        model_code = self.bernoulli_model_code
        t0 = time.time()
        model = StanModel(model_code=model_code)
        assert model is not None
        msg = "Compile time: {}s (vs. RStan 28s)\n".format(int(time.time()-t0))
        logging.info(msg)

    def test_bernoulli_sampling(self):
        fit = self.model.sampling(data=self.bernoulli_data)
        assert fit.sim['iter'] == 2000
        assert fit.sim['pars_oi'] == ['theta', 'lp__']
        assert len(fit.sim['samples']) == 4
        assert 0.1 < np.mean(fit.sim['samples'][0]['chains']['theta']) < 0.4
        assert 0.1 < np.mean(fit.sim['samples'][1]['chains']['theta']) < 0.4
        assert 0.1 < np.mean(fit.sim['samples'][2]['chains']['theta']) < 0.4
        assert 0.1 < np.mean(fit.sim['samples'][3]['chains']['theta']) < 0.4
        assert 0.01 < np.var(fit.sim['samples'][0]['chains']['theta']) < 0.02
        assert 0.01 < np.var(fit.sim['samples'][1]['chains']['theta']) < 0.02
        assert 0.01 < np.var(fit.sim['samples'][2]['chains']['theta']) < 0.02
        assert 0.01 < np.var(fit.sim['samples'][3]['chains']['theta']) < 0.02

    def test_bernoulli_extract(self):
        fit = self.model.sampling(data=self.bernoulli_data)
        extr = fit.extract(permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02

        # permuted=False
        extr = fit.extract(permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4

        # permuted=True
        extr = fit.extract('lp__', permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        extr = fit.extract('theta', permuted=True)
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02
        extr = fit.extract('theta', permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4
