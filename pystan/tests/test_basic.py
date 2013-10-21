import logging
import os
import tempfile
import time
import unittest

import numpy as np

from pystan import StanModel


class TestNormal(unittest.TestCase):

    model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
    model = StanModel(model_code=model_code, model_name="normal1",
                      verbose=True)

    def test_constructor(self):
        self.assertEqual(self.model.model_name, "normal1")

    def test_log_prob(self):
        fit = self.model.sampling()
        extr = fit.extract()
        y_last, log_prob_last = extr['y'][-1], extr['lp__'][-1]
        self.assertEqual(fit.log_prob(y_last), log_prob_last)


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

    fit = model.sampling(data=bernoulli_data)

    def test_bernoulli_constructor(self):
        model = self.model
        self.assertEqual(model.model_name, "bernoulli")
        self.assertTrue(model.model_cppname.endswith("bernoulli"))

    def test_bernoulli_sampling(self):
        fit = self.fit
        self.assertEqual(fit.sim['iter'], 2000)
        self.assertEqual(fit.sim['pars_oi'], ['theta', 'lp__'])
        self.assertEqual(len(fit.sim['samples']), 4)
        assert 0.1 < np.mean(fit.sim['samples'][0]['chains']['theta']) < 0.4
        assert 0.1 < np.mean(fit.sim['samples'][1]['chains']['theta']) < 0.4
        assert 0.1 < np.mean(fit.sim['samples'][2]['chains']['theta']) < 0.4
        assert 0.1 < np.mean(fit.sim['samples'][3]['chains']['theta']) < 0.4
        assert 0.01 < np.var(fit.sim['samples'][0]['chains']['theta']) < 0.02
        assert 0.01 < np.var(fit.sim['samples'][1]['chains']['theta']) < 0.02
        assert 0.01 < np.var(fit.sim['samples'][2]['chains']['theta']) < 0.02
        assert 0.01 < np.var(fit.sim['samples'][3]['chains']['theta']) < 0.02

    def test_bernoulli_sampling_error(self):
        bad_data = self.bernoulli_data.copy()
        del bad_data['N']
        try:
            assertRaisesRegex = self.assertRaisesRegex
        except AttributeError:
            assertRaisesRegex = self.assertRaisesRegexp
        with assertRaisesRegex(RuntimeError, 'variable does not exist'):
            fit = self.model.sampling(data=bad_data)

    def test_bernoulli_extract(self):
        fit = self.fit
        extr = fit.extract(permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02

        # permuted=False
        extr = fit.extract(permuted=False)
        self.assertEqual(extr.shape, (1000, 4, 2))
        self.assertTrue(0.1 < np.mean(extr[:, 0, 0]) < 0.4)

        # permuted=True
        extr = fit.extract('lp__', permuted=True)
        assert -7.4 < np.mean(extr['lp__']) < -7.0
        extr = fit.extract('theta', permuted=True)
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02
        extr = fit.extract('theta', permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4

    def test_bernoulli_summary(self):
        fit = self.fit
        s = fit.summary()
        assert s is not None
        repr(fit)
        print(fit)

    def test_bernoulli_plot(self):
        fit = self.fit
        fig = fit.plot()
        assert fig is not None

    def test_bernoulli_sampling_sample_file(self):
        tmpdir = tempfile.mkdtemp()
        sample_file = os.path.join(tmpdir, 'sampling.csv')
        sample_file_base = os.path.splitext(os.path.basename(sample_file))[0]
        fit = self.model.sampling(data=self.bernoulli_data, sample_file=sample_file)
        assert all([sample_file_base in fn for fn in os.listdir(tmpdir)])

        fit = self.model.sampling(data=self.bernoulli_data, sample_file='/tmp/doesnotexist')
        assert fit is not None

    # FIXME: not working right now -- need to debug
    # def test_bernoulli_optimizing_sample_file(self):
    #     tmpdir = tempfile.mkdtemp()
    #     sample_file = os.path.join(tmpdir, 'optim.csv')
    #     sample_file_base = os.path.splitext(os.path.basename(sample_file))[0]
    #     fit = self.model.optimizing(data=self.bernoulli_data, sample_file=sample_file)
    #     assert all([sample_file_base in fn for fn in os.listdir(tmpdir)])

    #     fit = self.model.optimizing(data=self.bernoulli_data, sample_file='/tmp/doesnotexist')
    #     assert fit is not None
