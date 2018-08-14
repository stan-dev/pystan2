import distutils.errors
import sys
import unittest

import pystan
from pystan._compat import PY2


class TestExtraCompileArgs(unittest.TestCase):

    def test_extra_compile_args(self):
        extra_compile_args = [
            '-O3',
            '-ftemplate-depth-1024',
            '-Wno-unused-function',
            '-Wno-uninitialized',
            '-std=c++11',
        ]
        if sys.platform.startswith("win"):
            extra_compile_args.extend([
                "-D_hypot=hypot",
                "-pthread",
                "-fexceptions"
            ])
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        model = pystan.StanModel(model_code=model_code, model_name="normal1",
                                 extra_compile_args=extra_compile_args)
        fit = model.sampling()
        extr = fit.extract()
        y_last, log_prob_last = extr['y'][-1], extr['lp__'][-1]
        self.assertEqual(fit.log_prob(y_last), log_prob_last)

    def test_extra_compile_args_failure(self):
        extra_compile_args = ['-non-existent-option']
        if sys.platform.startswith("win"):
            return
        model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
        assertRaisesRegex = self.assertRaisesRegexp if PY2 else self.assertRaisesRegex
        with assertRaisesRegex(distutils.errors.CompileError, 'failed with exit status'):
            pystan.StanModel(model_code=model_code, model_name="normal1",
                             extra_compile_args=extra_compile_args)
