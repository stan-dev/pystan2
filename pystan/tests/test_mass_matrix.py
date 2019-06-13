import unittest

import os
import numpy as np
import tempfile
import pystan
from pystan import StanModel
from pystan.tests.helper import get_model

class TestMassMatrix(unittest.TestCase):

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
            beta[k,d] ~ normal(if_else(d==2,100, 0),1);
        }"""
        cls.model = get_model("matrix_normal_model", model_code)
        #cls.model = StanModel(model_code=model_code)

        control = {"metric" : "diag_e"}
        fit = cls.model.sampling(chains=1, data=dict(K=3, D=4), warmup=1500, iter=1501, control=control, check_hmc_diagnostics=False, seed=14)
        cls.pos_def_matrix_diag_e = fit.get_mass_matrix()[0]
        cls.stepsize_diag_e = fit.get_stepsize()[0]

        control = {"metric" : "dense_e"}
        fit = cls.model.sampling(chains=1, data=dict(K=3, D=4), warmup=1500, iter=1501, control=control, check_hmc_diagnostics=False, seed=14)
        cls.pos_def_matrix_dense_e = fit.get_mass_matrix()[0]
        cls.stepsize_dense_e = fit.get_stepsize()[0]

    def test_get_mass_matrix(self):
        sm = self.model

        control_dict = {"metric" : "diag_e"}
        fit = sm.sampling(chains=3, data=dict(K=3, D=4), warmup=1000, iter=1001, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        mass_matrix = fit.get_mass_matrix()
        assert isinstance(mass_matrix, list)
        assert len(mass_matrix) == 3
        assert all(mm.shape == (3*4,) for mm in mass_matrix)

    def test_path_mass_matrix_diag_e(self):
        sm = self.model

        path = os.path.join(tempfile.mkdtemp(), "mass_matrix_t1.Rdata")
        mass_matrix = self.pos_def_matrix_diag_e
        stepsize = self.stepsize_diag_e
        pystan.misc.stan_rdump({"inv_metric" : mass_matrix}, path)
        control_dict = {"metric" : "diag_e", "mass_matrix" : path, "stepsize" : stepsize}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        mass_matrix_ = fit.get_mass_matrix()
        assert all(mm.shape == (3*4,) for mm in mass_matrix_)
        assert all(np.all(mass_matrix == mm) for mm in mass_matrix_)

        try:
            os.remove(path)
        except OSError:
            pass

    def test_path_mass_matrix_dict_diag_e(self):
        sm = self.model

        paths = []
        mass_matrices = []
        for i in range(4):
            path = os.path.join(tempfile.mkdtemp(), "mass_matrix_t2_{}.Rdata".format(i))
            mass_matrix = self.pos_def_matrix_diag_e
            pystan.misc.stan_rdump({"inv_metric" : mass_matrix}, path)
            paths.append(path)
            mass_matrices.append(mass_matrix)
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "diag_e", "mass_matrix" : dict(enumerate(paths)), "stepsize" : stepsize}
        fit = sm.sampling(chains=4, data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        mass_matrix_ = fit.get_mass_matrix()
        assert all(mm.shape == (3*4,) for mm in mass_matrix_)
        assert all(mm.shape == (3*4,) for mm in mass_matrices)
        assert all(np.all(mm_ == mm) for mm_, mm in zip(mass_matrix_, mass_matrices))

        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_path_mass_matrix_dense_e(self):
        sm = self.model

        path = os.path.join(tempfile.mkdtemp(), "mass_matrix_t3.Rdata")
        mass_matrix = self.pos_def_matrix_dense_e
        stepsize = self.stepsize_diag_e
        pystan.misc.stan_rdump({"inv_metric" : mass_matrix}, path)
        control_dict = {"metric" : "dense_e", "mass_matrix" : path, "stepsize" : stepsize}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        mass_matrix_ = fit.get_mass_matrix()
        assert all(mm.shape == (3*4,3*4) for mm in mass_matrix_)
        assert all(np.all(mass_matrix == mm) for mm in mass_matrix_)

        try:
            os.remove(path)
        except OSError:
            pass

    def test_one_mass_matrix_diag_e(self):
        sm = self.model

        mass_matrix = self.pos_def_matrix_diag_e
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "diag_e", "mass_matrix" : mass_matrix, "stepsize" : stepsize}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        mass_matrix_ = fit.get_mass_matrix()
        print(mass_matrix)
        print(mass_matrix_)
        assert all(mm.shape == (3*4,) for mm in mass_matrix_)
        assert all(np.all(mass_matrix == mm) for mm in mass_matrix_)

    def test_one_mass_matrix_dense_e(self):
        sm = self.model

        mass_matrix = self.pos_def_matrix_dense_e
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "dense_e", "mass_matrix" : mass_matrix, "stepsize" : stepsize}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

        mass_matrix_ = fit.get_mass_matrix()
        assert all(mm.shape == (3*4,3*4) for mm in mass_matrix_)
        assert all(np.all(mass_matrix == mm) for mm in mass_matrix_)

    def test_mass_matrix_nuts_diag_e_adapt_true(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : True}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()

        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "diag_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,)
            assert mass_matrix2.shape == (3*4,)
            assert np.all(mass_matrix1 == mass_matrix2)

    def test_mass_matrix_nuts_diag_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()

        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "diag_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,)
            assert mass_matrix2.shape == (3*4,)
            assert np.all(mass_matrix1 == mass_matrix2)

    def test_mass_matrix_nuts_dense_e_adapt_true(self):
        sm = self.model

        control_dict = {"metric" : "dense_e", "adapt_engaged" : True}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()

        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "dense_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,3*4)
            assert mass_matrix2.shape == (3*4,3*4)
            assert np.all(mass_matrix1 == mass_matrix2)

    def test_mass_matrix_nuts_dense_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "dense_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()

        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "dense_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,3*4)
            assert mass_matrix2.shape == (3*4,3*4)
            assert np.all(mass_matrix1 == mass_matrix2)

    def test_mass_matrix_hmc_diag_e_adapt_true(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : True}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()

        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "diag_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,)
            assert mass_matrix2.shape == (3*4,)
            assert np.all(mass_matrix1 == mass_matrix2)

    def test_mass_matrix_hmc_diag_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "diag_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()

        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "diag_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,)
            assert mass_matrix2.shape == (3*4,)
            assert np.all(mass_matrix1 == mass_matrix2)

    def test_mass_matrix_hmc_dense_e_adapt_true(self):
        sm = self.model

        # this is fragile, using precalculated mass_matrix and stepsizes
        mass_matrix = self.pos_def_matrix_dense_e
        stepsize = self.stepsize_diag_e
        control_dict = {"metric" : "dense_e", "adapt_engaged" : True, "mass_matrix" : mass_matrix, "stepsize" : stepsize}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()


        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "dense_e", "adapt_engaged" : True, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,3*4)
            assert mass_matrix2.shape == (3*4,3*4)
            assert np.all(mass_matrix1 == mass_matrix2)

    def test_mass_matrix_hmc_dense_e_adapt_false(self):
        sm = self.model

        control_dict = {"metric" : "dense_e", "adapt_engaged" : False}
        fit1 = sm.sampling(data=dict(K=3, D=4), warmup=1500, iter=1501, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit1 = fit1.get_mass_matrix()
        stepsize = fit1.get_stepsize()

        control_dict = {"mass_matrix" : dict(enumerate(mass_matrix_fit1)), "metric" : "dense_e", "adapt_engaged" : False, "stepsize" : stepsize}
        fit2 = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="HMC", control=control_dict, check_hmc_diagnostics=False, seed=14)
        mass_matrix_fit2 = fit2.get_mass_matrix()

        for mass_matrix1, mass_matrix2 in zip(mass_matrix_fit1, mass_matrix_fit2):
            assert mass_matrix1.shape == (3*4,3*4)
            assert mass_matrix2.shape == (3*4,3*4)
            assert np.all(mass_matrix1 == mass_matrix2)

    @unittest.expectedFailure
    def test_fail_diag_e(self):
        sm = self.model

        mass_matrix = np.ones((3*4,3*4))/10+np.eye(3*4)
        control_dict = {"metric" : "diag_e", "mass_matrix" : mass_matrix}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)

    @unittest.expectedFailure
    def test_fail_dense_e(self):
        sm = self.model

        mass_matrix = self.ones(3*4)
        control_dict = {"metric" : "dense_e", "mass_matrix" : mass_matrix}
        fit = sm.sampling(data=dict(K=3, D=4), warmup=0, iter=10, algorithm="NUTS", control=control_dict, check_hmc_diagnostics=False, seed=14)
