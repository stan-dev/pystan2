import os
import pickle
import sys
import tempfile
import unittest

import pystan


class TestPickle(unittest.TestCase):

    def test_pickle_model(self):
        tmpdir = tempfile.mkdtemp()
        pickle_file = os.path.join(tmpdir, 'stanmodel.pkl')
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        m = pystan.StanModel(model_code=model_code, model_name="normal1",
                             save_dso=False)
        module_name = m.module.__name__
        with open(pickle_file, 'wb') as f:
            pickle.dump(m, f)
        del m
        del sys.modules[module_name]

        with open(pickle_file, 'rb') as f:
            m = pickle.load(f)
        assert m.model_name.startswith("normal1")

        m = pystan.StanModel(model_code=model_code, model_name="normal2")
        module_name = m.module.__name__
        module_filename = m.module.__file__
        with open(pickle_file, 'wb') as f:
            pickle.dump(m, f)
        del m
        del sys.modules[module_name]

        with open(pickle_file, 'rb') as f:
            m = pickle.load(f)
        assert m.model_name.startswith("normal2")
        assert m.module is not None
        assert module_filename != m.module.__file__
        fit = m.sampling()
        y = fit.extract()['y']
        assert len(y) == 4000

    def test_pickle_fit(self):
        tmpdir = tempfile.mkdtemp()
        num_iter = 100
        pickle_file = os.path.join(tmpdir, 'stanfit.pkl')
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'

        sm = pystan.StanModel(model_code=model_code, model_name="normal1")
        fit = sm.sampling(iter=num_iter)
        y = fit.extract()['y'].copy()

        # pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(fit, f)
        del fit

        # load from file
        with open(pickle_file, 'rb') as f:
            fit = pickle.load(f)

        self.assertIsNotNone(fit)
        self.assertTrue((fit.extract()['y'] == y).all())
