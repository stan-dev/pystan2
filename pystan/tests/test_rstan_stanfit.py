import unittest

from pystan import StanModel
from pystan._compat import PY2


class TestStanfit(unittest.TestCase):

    def test_init_zero_exception_inf_grad(self):
        code = """
        parameters {
            real x;
        }
        model {
            lp__ <- 1 / log(x);
        }
        """
        sm = StanModel(model_code=code)
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(RuntimeError, 'divergent gradient'):
            sm.sampling(init='0', iter=1)
