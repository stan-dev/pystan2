import unittest

import os
import numpy as np
import tempfile
import pystan
from pystan import StanModel
from pystan.tests.helper import get_model

class TestEuclidianMetric(unittest.TestCase):

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
        }"""
        cls.model = get_model("matrix_normal_model", model_code)
        #cls.model = StanModel(model_code=model_code)

        control = {"metric" : "diag_e"}
        fit = cls.model.sampling(chains=1, data=dict(K=3, D=4), warmup=1500, iter=1501, control=control, check_hmc_diagnostics=False, seed=14)
        cls.pos_def_matrix_diag_e = fit.get_inv_metric()[0]
        cls.stepsize_diag_e = fit.get_stepsize()[0]

        control = {"metric" : "dense_e"}
        fit = cls.model.sampling(chains=1, data=dict(K=3, D=4), warmup=1500, iter=1501, control=control, check_hmc_diagnostics=False, seed=14)
        cls.pos_def_matrix_dense_e = fit.get_inv_metric()[0]
        cls.stepsize_dense_e = fit.get_stepsize()[0]

    def test_get_inv_metric(self):
        sm = self.model

        control_dict = {"metric" : "diag_e"}
        fit = sm.sampling(chains=3, data=dict(K=3, D=4), warmup=1000, iter=1001, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        inv_metric = fit.get_inv_metric()
        assert isinstance(inv_metric, list)
        assert len(inv_metric) == 3
        assert all(mm.shape == (3*4,) for mm in inv_metric)

    def test_path_inv_metric_diag_e(self):
        sm = self.model

        path = os.path.join(tempfile.mkdtemp(), "inv_metric_t1.Rdata")
        inv_metric = self.pos_def_matrix_diag_e
        stepsize = self.stepsize_diag_e
        pystan.misc.stan_rdump({"inv_metric" : inv_metric}, path)
        control_dict = {"metric" : "diag_e", "inv_metric" : path, "stepsize" : stepsize, "adapt_engaged" : False}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        inv_metric_ = fit.get_inv_metric()
        assert all(mm.shape == (3*4,) for mm in inv_metric_)
        assert all(np.all(inv_metric == mm) for mm in inv_metric_)

        try:
            os.remove(path)
        except OSError:
            pass

    def test_path_inv_metric_dict_diag_e(self):
        sm = self.model

        paths = []
        inv_metrics = []
        for i in range(4):
            path = os.path.join(tempfile.mkdtemp(), "inv_metric_t2_{}.Rdata".format(i))
            inv_metric = self.pos_def_matrix_diag_e
            pystan.misc.stan_rdump({"inv_metric" : inv_metric}, path)
            paths.append(path)
            inv_metrics.append(inv_metric)
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "diag_e", "inv_metric" : dict(enumerate(paths)), "stepsize" : stepsize, "adapt_engaged" : False}
        fit = sm.sampling(chains=4, data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        inv_metric_ = fit.get_inv_metric()
        assert all(mm.shape == (3*4,) for mm in inv_metric_)
        assert all(mm.shape == (3*4,) for mm in inv_metrics)
        assert all(np.all(mm_ == mm) for mm_, mm in zip(inv_metric_, inv_metrics))

        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_path_inv_metric_dense_e(self):
        sm = self.model

        path = os.path.join(tempfile.mkdtemp(), "inv_metric_t3.Rdata")
        inv_metric = self.pos_def_matrix_dense_e
        stepsize = self.stepsize_diag_e
        pystan.misc.stan_rdump({"inv_metric" : inv_metric}, path)
        control_dict = {"metric" : "dense_e", "inv_metric" : path, "stepsize" : stepsize, "adapt_engaged" : False}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        inv_metric_ = fit.get_inv_metric()
        assert all(mm.shape == (3*4,3*4) for mm in inv_metric_)
        assert all(np.all(inv_metric == mm) for mm in inv_metric_)

        try:
            os.remove(path)
        except OSError:
            pass

    def test_one_inv_metric_diag_e(self):
        sm = self.model

        inv_metric = self.pos_def_matrix_diag_e
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "diag_e", "inv_metric" : inv_metric, "stepsize" : stepsize, "adapt_engaged" : False}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        inv_metric_ = fit.get_inv_metric()

        assert all(mm.shape == (3*4,) for mm in inv_metric_)
        assert all(np.all(inv_metric == mm) for mm in inv_metric_)

    def test_one_inv_metric_dense_e(self):
        sm = self.model

        inv_metric = self.pos_def_matrix_dense_e
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "dense_e", "inv_metric" : inv_metric, "stepsize" : stepsize, "adapt_engaged" : False}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        inv_metric_ = fit.get_inv_metric()
        assert all(mm.shape == (3*4,3*4) for mm in inv_metric_)
        assert all(np.all(inv_metric == mm) for mm in inv_metric_)

    def test_inv_metric_nuts_diag_e_adapt_true(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : True}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()

        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "diag_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=10, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,)
            assert inv_metric2.shape == (3*4,)

    def test_inv_metric_nuts_diag_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()

        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "diag_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,)
            assert inv_metric2.shape == (3*4,)
            assert np.all(inv_metric1 == inv_metric2)

    def test_inv_metric_nuts_dense_e_adapt_true(self):
        sm = self.model

        control_dict = {"metric" : "dense_e", "adapt_engaged" : True}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()

        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "dense_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=10, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,3*4)
            assert inv_metric2.shape == (3*4,3*4)

    def test_inv_metric_nuts_dense_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "dense_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()

        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "dense_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,3*4)
            assert inv_metric2.shape == (3*4,3*4)
            assert np.all(inv_metric1 == inv_metric2)

    def test_inv_metric_hmc_diag_e_adapt_true(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : True}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()

        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "diag_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=10, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,)
            assert inv_metric2.shape == (3*4,)

    def test_inv_metric_hmc_diag_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()

        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "diag_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,)
            assert inv_metric2.shape == (3*4,)
            assert np.all(inv_metric1 == inv_metric2)

    def test_inv_metric_hmc_dense_e_adapt_true(self):
        sm = self.model

        # this is fragile, using precalculated inv_metric and stepsizes
        inv_metric = self.pos_def_matrix_dense_e
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "dense_e", "adapt_engaged" : True, "inv_metric" : inv_metric, "stepsize" : stepsize}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=10, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()


        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "dense_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=10, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,3*4)
            assert inv_metric2.shape == (3*4,3*4)

    def test_inv_metric_hmc_dense_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "dense_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit1 = fit1.get_inv_metric()
        stepsize = fit1.get_stepsize()

        control_dict = {"inv_metric" : dict(enumerate(inv_metric_fit1)), "metric" : "dense_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        inv_metric_fit2 = fit2.get_inv_metric()

        for inv_metric1, inv_metric2 in zip(inv_metric_fit1, inv_metric_fit2):
            assert inv_metric1.shape == (3*4,3*4)
            assert inv_metric2.shape == (3*4,3*4)
            assert np.all(inv_metric1 == inv_metric2)

    @unittest.expectedFailure
    def test_fail_diag_e(self):
        sm = self.model

        inv_metric = np.ones((3*4,3*4))/10+np.eye(3*4)
        control_dict = {"metric" : "diag_e", "inv_metric" : inv_metric}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

    @unittest.expectedFailure
    def test_fail_dense_e(self):
        sm = self.model

        inv_metric = self.ones(3*4)
        control_dict = {"metric" : "dense_e", "inv_metric" : inv_metric}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
