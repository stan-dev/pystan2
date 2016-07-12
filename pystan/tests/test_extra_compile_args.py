import unittest

import pystan


class TestExtraCompileArgs(unittest.TestCase):

    def test_extra_compile_args(self):
        extra_compile_args = [
            '-O3',
            '-ftemplate-depth-1024',
            '-Wno-unused-function',
            '-Wno-uninitialized',
        ]
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        model = pystan.StanModel(model_code=model_code, model_name="normal1",
                                 extra_compile_args=extra_compile_args)
        fit = model.sampling()
        extr = fit.extract()
        y_last, log_prob_last = extr['y'][-1], extr['lp__'][-1]
        self.assertEqual(fit.log_prob(y_last), log_prob_last)
