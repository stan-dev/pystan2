import unittest

import numpy as np

import pystan
from pystan.tests.helper import get_model
from pystan.experimental import EvalLogProb

class TestOptimizingExample(unittest.TestCase):
    """Test optimizing example from documentation"""

    @classmethod
    def setUpClass(cls):
        ocode = """
        data {
            int<lower=1> N;
            real y[N];
        }
        parameters {
            real mu;
        }
        model {
            y ~ normal(mu, 1);
        }
        """

        cls.sm = get_model("optimize_normal", model_code=ocode)

    def test_optimizing(self):
        sm = self.sm
        np.random.seed(3)
        y2 = np.random.normal(size=20)
        op = sm.optimizing(data=dict(y=y2, N=len(y2)))
        self.assertAlmostEqual(op['mu'], np.mean(y2))


class TestExperimentalEvalLogProb(unittest.TestCase):
    """Test evaluation of the LogProb and the LogProbGrad without sampling-step."""

    @classmethod
    def setUpClass(cls):
        ocode = """
        data {
            int<lower=1> N;
            real y[N];
        }
        parameters {
            real mu;
        }
        model {
            y ~ normal(mu, 1);
        }
        """

        cls.sm = get_model("optimize_normal", model_code=ocode)

    def test_logprob_from_model(self):
        func_log_prob = EvalLogProb({"y" : [0], "N" : 1}, model=self.sm)
        log_prob, grad_log_prob = func_log_prob({"mu" : 0.0})
        self.assertEqual(log_prob, 0.0)
        self.assertEqual(grad_log_prob[0], 0.0)

        log_prob, grad_log_prob = func_log_prob({"mu" : -3.0})
        self.assertEqual(log_prob, -4.5)
        self.assertEqual(grad_log_prob[0], 3.0)

        log_prob, grad_log_prob = func_log_prob({"mu" : 2.0})
        self.assertEqual(log_prob, -2.0)
        self.assertEqual(grad_log_prob[0], -2.0)

    def test_logprob_from_fit(self):
        fit = self.sm.sampling(data={"y" : [0], "N" : 1}, chains=1, warmup=1, iter=2, check_hmc_diagnostics=False)
        func_log_prob = EvalLogProb({"y" : [0], "N" : 1}, fit=fit)
        log_prob, grad_log_prob = func_log_prob({"mu" : 0.0})
        self.assertEqual(log_prob, 0.0)
        self.assertEqual(grad_log_prob[0], 0.0)

        log_prob, grad_log_prob = func_log_prob({"mu" : -3.0})
        self.assertEqual(log_prob, -4.5)
        self.assertEqual(grad_log_prob[0], 3.0)

        log_prob, grad_log_prob = func_log_prob({"mu" : 2.0})
        self.assertEqual(log_prob, -2.0)
        self.assertEqual(grad_log_prob[0], -2.0)
