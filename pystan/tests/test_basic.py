from collections import OrderedDict
import gc
import os
import tempfile
import unittest

import numpy as np

import pystan
from pystan._compat import PY2


class TestNormal(unittest.TestCase):

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

    def test_control_stepsize(self):
        fit = self.model.sampling(control=dict(stepsize=0.001))
        self.assertIsNotNone(fit)


class TestBernoulli(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
        cls.bernoulli_data = bernoulli_data = {'N': 10, 'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}

        cls.model = model = pystan.StanModel(model_code=bernoulli_model_code, model_name="bernoulli")

        cls.fit = model.sampling(data=bernoulli_data)

    def test_bernoulli_constructor(self):
        model = self.model
        # obfuscate_model_name is True
        self.assertNotEqual(model.model_name, "bernoulli")
        self.assertTrue(model.model_name.startswith("bernoulli"))
        self.assertTrue(model.model_cppname.startswith("bernoulli"))

    def test_bernoulli_OrderedDict(self):
        data = OrderedDict(self.bernoulli_data.items())
        self.model.sampling(data=data, iter=2)

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
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(RuntimeError, 'variable does not exist'):
            fit = self.model.sampling(data=bad_data)

    def test_bernoulli_extract(self):
        fit = self.fit
        extr = fit.extract(permuted=True)
        assert -7.9 < np.mean(extr['lp__']) < -7.0, np.mean(extr['lp__'])
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02

        # use __getitem__
        assert -7.9 < np.mean(fit['lp__']) < -7.0, np.mean(fit['lp__'])
        assert 0.1 < np.mean(fit['theta']) < 0.4
        assert 0.01 < np.var(fit['theta']) < 0.02

        # permuted=False
        extr = fit.extract(permuted=False)
        self.assertEqual(extr.shape, (1000, 4, 2))
        self.assertTrue(0.1 < np.mean(extr[:, 0, 0]) < 0.4)

        # permuted=True
        extr = fit.extract('lp__', permuted=True)
        assert -7.9 < np.mean(extr['lp__']) < -7.0
        extr = fit.extract('theta', permuted=True)
        assert 0.1 < np.mean(extr['theta']) < 0.4
        assert 0.01 < np.var(extr['theta']) < 0.02
        extr = fit.extract('theta', permuted=False)
        assert extr.shape == (1000, 4, 2)
        assert 0.1 < np.mean(extr[:, 0, 0]) < 0.4

    def test_bernoulli_random_seed_consistency(self):
        thetas = []
        for _ in range(2):
            fit = self.model.sampling(data=self.bernoulli_data, seed=42)
            thetas.append(fit.extract('theta', permuted=True)['theta'])
        np.testing.assert_equal(*thetas)

    def test_bernoulli_random_seed_inconsistency(self):
        thetas = []
        for seed in range(2):
            # seeds will be 0, 1
            fit = self.model.sampling(data=self.bernoulli_data,
                                      seed=np.random.RandomState(seed))
            thetas.append(fit.extract('theta', permuted=True)['theta'])
        self.assertFalse(np.allclose(*thetas))

    def test_bernoulli_summary(self):
        fit = self.fit
        s = fit.summary()
        assert s is not None
        # printing to make sure no exception raised
        repr(fit)
        print(fit)

    def test_bernoulli_plot(self):
        fit = self.fit
        fig = fit.plot()
        assert fig is not None
        fig = fit.plot(['theta'])
        assert fig is not None
        fig = fit.plot('theta')
        assert fig is not None

    def test_bernoulli_sampling_sample_file(self):
        tmpdir = tempfile.mkdtemp()
        sample_file = os.path.join(tmpdir, 'sampling.csv')
        fit = self.model.sampling(data=self.bernoulli_data, sample_file=sample_file)

        num_chains = len(fit.sim['samples'])
        for i in range(num_chains):
            fn = os.path.splitext(os.path.join(tmpdir, 'sampling.csv'))[0] + "_{}.csv".format(i)
            assert os.path.exists(fn)

        fit = self.model.sampling(data=self.bernoulli_data, sample_file='/tmp/pathdoesnotexist/sampling.csv')
        assert fit is not None

    def test_bernoulli_optimizing_sample_file(self):
        tmpdir = tempfile.mkdtemp()
        sample_file = os.path.join(tmpdir, 'optim.csv')
        self.model.optimizing(data=self.bernoulli_data, sample_file=sample_file)
        assert os.path.exists(sample_file)

        fit = self.model.optimizing(data=self.bernoulli_data, sample_file='/tmp/pathdoesnotexist/optim.csv')
        assert fit is not None
