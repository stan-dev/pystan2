import unittest

import numpy as np

import pystan


class TestFixedParam(unittest.TestCase):

    def test_fixedp(self):
        code = """
            model {
            }

            generated quantities {
            real y;
            y <- normal_rng(0, 1);
            }
            """
        fit = stan(model_code=code)
        fit2 = stan(fit=fit, algorithm='Fixed_param')
