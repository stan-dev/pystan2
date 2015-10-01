import os
import pickle
import sys
import tempfile
import unittest

import pystan


class TestPickle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tmpdir = tempfile.mkdtemp()
        cls.pickle_file = os.path.join(tmpdir, 'stanmodel.pkl')
        cls.model_code = 'parameters {real y;} model {y ~ normal(0,1);}'

    def test_pickle_model(self):
        pickle_file = self.pickle_file
        model_code = self.model_code
        m = pystan.StanModel(model_code=model_code, model_name="normal2")
        module_name = m.module.__name__
        module_filename = m.module.__file__
        with open(pickle_file, 'wb') as f:
            pickle.dump(m, f)
        del m
        del sys.modules[module_name]

        with open(pickle_file, 'rb') as f:
            m = pickle.load(f)
        self.assertTrue(m.model_name.startswith("normal2"))
        self.assertIsNotNone(m.module)
        self.assertNotEqual(module_filename, m.module.__file__)
        fit = m.sampling()
        y = fit.extract()['y']
        assert len(y) == 4000

    def test_pickle_fit(self):
        num_iter = 100
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'

        sm = pystan.StanModel(model_code=model_code, model_name="normal1")

        # additional error checking
        fit = sm.sampling(iter=num_iter)
        y = fit.extract()['y'].copy()
        self.assertIsNotNone(y)

        # pickle
        module_name = sm.module.__name__
        # access to tmp directory with travis ci often fails
        try:
            pickled_model = pickle.dumps(sm)
            pickled_fit = pickle.dumps(fit)
        except RuntimeError:
            pickled_model = None
            pickled_fit = None
        else:
            del sm
            del fit

        # unload module
        if module_name in sys.modules:
            del(sys.modules[module_name])

        # load from file
        if pickled_model and pickled_fit:
            sm_from_pickle = pickle.loads(pickled_model)
            fit_from_pickle = pickle.loads(pickled_fit)
            self.assertIsNotNone(fit_from_pickle)
            self.assertTrue((fit_from_pickle.extract()['y'] == y).all())
