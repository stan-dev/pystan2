import io
import os
import pickle
import sys
import tempfile
import unittest
import numpy as np

import pystan
from pystan.tests.helper import get_model
from pystan.experimental import unpickle_fit

class TestPickle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pickle_file = os.path.join(tempfile.mkdtemp(), 'stanmodel.pkl')
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
        if not sys.platform.startswith('win'):
            # will fail on Windows
            self.assertNotEqual(module_filename, m.module.__file__)
        fit = m.sampling()
        y = fit.extract()['y']
        assert len(y) == 4000

    def test_pickle_fit(self):
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'

        sm = pystan.StanModel(model_code=model_code, model_name="normal1")

        # additional error checking
        fit = sm.sampling(iter=100)
        y = fit.extract()['y'].copy()
        self.assertIsNotNone(y)

        # pickle
        pickled_model = pickle.dumps(sm)
        module_name = sm.module.__name__
        del sm
        pickled_fit = pickle.dumps(fit)
        del fit

        # unload module
        if module_name in sys.modules:
            del(sys.modules[module_name])

        # load from file
        sm_from_pickle = pickle.loads(pickled_model)
        fit_from_pickle = pickle.loads(pickled_fit)
        self.assertIsNotNone(fit_from_pickle)
        self.assertTrue((fit_from_pickle.extract()['y'] == y).all())

    def test_pickle_model_and_reload(self):
        pickle_file = self.pickle_file
        pickle_file2 = os.path.join(tempfile.mkdtemp(), 'stanmodel.pkl')
        model_code = self.model_code
        model = pystan.StanModel(model_code=model_code, model_name="normal1")
        with open(pickle_file, 'wb') as f:
            pickle.dump(model, f)
        with open(pickle_file2, 'wb') as f:
            pickle.dump(model, f)

        del model

        with open(pickle_file, 'rb') as f:
            model_from_pickle = pickle.load(f)
        self.assertIsNotNone(model_from_pickle.sampling(iter=100).extract())
        with open(pickle_file2, 'rb') as f:
            model_from_pickle = pickle.load(f)
        self.assertIsNotNone(model_from_pickle.sampling(iter=100).extract())

    def test_model_unique_names(self):
        model_code = self.model_code
        model1 = pystan.StanModel(model_code=model_code, model_name="normal1")
        model2 = pystan.StanModel(model_code=model_code, model_name="normal1")
        self.assertNotEqual(model1.module_name, model2.module_name)

class TestPickleFitOnly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        model = get_model("standard_normal_model",
                          model_code, model_name="normal1",
                          verbose=True, obfuscate_model_name=False)
        fit = model.sampling()
        tempfolder = tempfile.mkdtemp()
        cls.pickle_fit = os.path.join(tempfolder, 'stanfit.pkl')
        cls.pickle_extract = os.path.join(tempfolder, 'stanextract.pkl')
        with io.open(cls.pickle_fit, mode="wb") as f:
            pickle.dump(fit, f)

        with io.open(cls.pickle_extract, mode="wb") as f:
            pickle.dump(fit.extract(), f)

        module_name = model.module.__name__
        del model
        del sys.modules[module_name]

    @unittest.expectedFailure
    def test_unpickle_fit_fail(self):
        with io.open(self.pickle_file, "rb") as f:
            pickle.load(f)

    def test_load_fit(self):
        fit, model = unpickle_fit(self.pickle_fit, open_func=io.open, open_kwargs={"mode" : "rb"}, return_model=True)
        self.assertIsNotNone(fit)
        self.assertIsNotNone(model)
        self.assertIsNotNone(fit.extract())
        self.assertTrue("y" in fit.extract())

        with io.open(self.pickle_extract, "rb") as f:
            extract = pickle.load(f)
        self.assertTrue(np.all(fit.extract()["y"] == extract["y"]))
