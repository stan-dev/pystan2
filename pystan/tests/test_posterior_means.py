import unittest

import numpy as np

import pystan


class TestPosteriorMeans(unittest.TestCase):
    """Test get_posterior_mean"""

    @classmethod
    def setUpClass(cls):
        model_code = """
        data {
            int<lower=0> n_subjects; // number of subjects
            int<lower=0,upper=40> score[n_subjects]; // UPSIT scores
            int<lower=0,upper=40> n_questions; // number of questions
        }
        parameters {
            real rating[n_subjects];
        }
        transformed parameters {
            real p[n_subjects];
            for (j in 1:n_subjects) {
                p[j] = exp(rating[j])/(1+exp(rating[j])); // Logistic function
                }
        }
        model {
            rating ~ normal(0, 1);
            score ~ binomial(40, p);
        }
        """

        cls.sm = pystan.StanModel(model_code=model_code)
        cls.data = {'n_questions': 40,
                    'n_subjects': 92,
                    'score': np.random.randint(0, 40, 92)}

    def test_posterior_means(self):
        sm = self.sm
        data = self.data
        fit = sm.sampling(data, iter=100)
        self.assertIsNotNone(fit.get_posterior_mean())
