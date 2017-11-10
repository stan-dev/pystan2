import unittest
import numpy as np
import pystan

class TestLookup(unittest.TestCase):
    def test_lookup(self):
        cbind = pystan.lookup("R.cbind")["StanFunction"]
        hstack = pystan.lookup("numpy.hstack")["StanFunction"]
        cbind_err = pystan.lookup("R.cbindd")["StanFunction"]
        cbind_2 = pystan.lookup("R.cbind", 1.0)["StanFunction"]
        hstack_2 = pystan.lookup("numpy.hstack", 1.0)["StanFunction"]
        for i in range(len(cbind)):
            self.assertEqual(cbind[i], "append_col")
            self.assertEqual(cbind[i], hstack[i])
            self.assertEqual(cbind[i], cbind_err[i])
            self.assertEqual(cbind[i], cbind_2[i])
            self.assertEqual(cbind[i], hstack_2[i])
        normal = pystan.lookup("scipy.stats.norm")["StanFunction"]
        for i in range(len(normal)):
            self.assertEqual("normal", normal[i][:6])
        poisson = pystan.lookup("scipy.stats.poisson")["StanFunction"]
        for i in range(len(poisson)):
            self.assertEqual("poisson", poisson[i][:7])
        wrongf = pystan.lookup("someverycrazyfunctionyouwontfind", 1.0)
        self.assertTrue(wrongf is None)
