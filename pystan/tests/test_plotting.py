import unittest
import numpy as np
import pystan
from matplotlib.pyplot import close

class TestPlotting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real mu;vector<lower=0>[4] tau;matrix[3,3] theta;} model {mu ~ normal(0,1);tau ~ student_t(3,0,5);for (row in 1:3) to_vector(theta[row]) ~ normal(0,1);}'
        cls.model = pystan.StanModel(model_code=model_code, model_name="test_plotting",
                                     verbose=True, obfuscate_model_name=False)
        cls.fit = cls.model.sampling(iter=200)

    def test_plot(self):
        fit = self.fit

        fig = fit.plot()
        assert fig is not None

        fig = fit.plot(['mu'])
        assert fig is not None

        fig = fit.plot('mu')
        assert fig is not None

        fig = fit.plot(['tau'])
        assert fig is not None

        fig = fit.plot(['theta'])
        assert fig is not None

        close('all')

        fig = fit.plot(['mu', 'tau'])
        assert fig is not None

        fig = fit.plot(kind='trace')
        assert fig is not None

        fig = fit.plot(kind='forest')
        assert fig is not None

        fig = fit.plot(kind='mcmc_parcoord')
        assert fig is not None

        fig = pystan.plot_fit(fit, kind='trace')
        assert fig is not None

        close('all')

        fig = pystan.plot_fit(fit, kind='forest')
        assert fig is not None

        fig = pystan.plot_fit(fit, kind='mcmc_parcoord')
        assert fig is not None

        close('all')

    def test_traceplot(self):
        fit = self.fit

        fig = fit.plot_traceplot()
        assert fig is not None

        fig = fit.plot_traceplot(['theta'])
        assert fig is not None

        fig = pystan.plot_traceplot(fit)
        assert fig is not None

        fig = pystan.plot_traceplot(fit, ['theta'])
        assert fig is not None

        fig = pystan.plot_traceplot(fit, ['theta'], dtypes={'theta' : int})
        assert fig is not None

        close('all')

    def test_forest(self):
        fit = self.fit

        fig = fit.plot_traceplot()
        assert fig is not None

        fig = fit.plot_forestplot(['theta'])
        assert fig is not None

        fig = pystan.plot_forestplot(fit)
        assert fig is not None

        fig = pystan.plot_forestplot(fit, ['theta'])
        assert fig is not None

        close('all')

    def test_mcmcparcoord(self):
        fit = self.fit

        fig = fit.plot_mcmc_parcoord()
        assert fig is not None

        fig = pystan.plot_mcmc_parcoord(fit)
        assert fig is not None

        close('all')
