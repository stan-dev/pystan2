import os
import tempfile
import unittest

import numpy as np

import pystan

class TestRStanZeroLen(unittest.TestCase):

    def test_zerolen(self):
        tempdir = tempfile.mkdtemp()
        csv_fname = os.path.join(tempdir, 'zerolen.csv')
        model_code = """
            data {
            int<lower=0> n;
            }
            parameters {
            real a0;
            real a[n];
            }
            model {
            a0 ~ normal(0,1);
            a ~ normal(a0,1);
            }
        """
        data = dict(n=0)
        fit = pystan.stan(model_code=model_code, model_name='normal1',
                        data=data, sample_file=csv_fname)
        pmean = fit.get_posterior_mean()[:, 0]
        summary_colnames = fit.summary()['summary_colnames']
        mean_ix = summary_colnames.index('mean')
        pmean2 = fit.summary()['c_summary'][:, mean_ix, 0]
        np.testing.assert_almost_equal(pmean, pmean2)
