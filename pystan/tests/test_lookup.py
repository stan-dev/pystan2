import unittest
import numpy as np
import pystan
import re

class TestLookup(unittest.TestCase):
    def test_lookup(self):
        cbind = pystan.lookup("R.cbind")["StanFunction"]
        hstack = pystan.lookup("numpy.hstack")["StanFunction"]
        for i in range(len(cbind)):
            self.assertEqual(str(cbind[i]), "append_col")
            self.assertEqual(cbind[i], hstack[i])
        normal = pystan.lookup("scipy.stats.norm")["StanFunction"]
        for i in range(len(normal)):
          self.assertTrue(re.fullmatch(r"^normal.*", normal[i]))
        poisson = pystan.lookup("scipy.stats.poisson")["StanFunction"]
        for i in range(len(poisson)):
          self.assertTrue(re.fullmatch(r"^poisson.*", poisson[i]))
