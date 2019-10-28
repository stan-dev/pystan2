import unittest

import numpy as np
from pandas.util.testing import assert_series_equal
from numpy.testing import assert_array_equal

import pystan


class TestExtract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ex_model_code = '''
        parameters {
            real alpha[2,3];
            real beta[2];
        }
        model {
            for (i in 1:2) for (j in 1:3)
            alpha[i, j] ~ normal(0, 1);
            for (i in 1:2)
            beta ~ normal(0, 2);
        }
        '''
        cls.sm = sm = pystan.StanModel(model_code=ex_model_code)
        cls.fit = sm.sampling(chains=4, iter=2000)

    def test_extract_permuted(self):
        ss = self.fit.extract(permuted=True)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        self.assertEqual(sorted(ss.keys()), sorted({'alpha', 'beta', 'lp__'}))
        self.assertEqual(alpha.shape, (4000, 2, 3))
        self.assertEqual(beta.shape, (4000, 2))
        self.assertEqual(lp__.shape, (4000,))
        self.assertTrue((~np.isnan(alpha)).all())
        self.assertTrue((~np.isnan(beta)).all())
        self.assertTrue((~np.isnan(lp__)).all())

        # extract one at a time
        alpha2 = self.fit.extract('alpha', permuted=True)['alpha']
        self.assertEqual(alpha2.shape, (4000, 2, 3))
        np.testing.assert_array_equal(alpha, alpha2)
        beta = self.fit.extract('beta', permuted=True)['beta']
        self.assertEqual(beta.shape, (4000, 2))
        lp__ = self.fit.extract('lp__', permuted=True)['lp__']
        self.assertEqual(lp__.shape, (4000,))

    def test_extract_permuted_false(self):
        fit = self.fit
        ss = fit.extract(permuted=False)
        num_samples = fit.sim['iter'] - fit.sim['warmup']
        self.assertEqual(ss.shape, (num_samples, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())

    def test_extract_permuted_false_pars(self):
        fit = self.fit
        ss = fit.extract(pars=['beta'], permuted=False)
        num_samples = fit.sim['iter'] - fit.sim['warmup']
        self.assertEqual(ss['beta'].shape, (num_samples, 4, 2))
        self.assertTrue((~np.isnan(ss['beta'])).all())

    def test_extract_permuted_false_pars_inc_warmup(self):
        fit = self.fit
        ss = fit.extract(pars=['beta'], inc_warmup=True, permuted=False)
        num_samples = fit.sim['iter']
        self.assertEqual(ss['beta'].shape, (num_samples, 4, 2))
        self.assertTrue((~np.isnan(ss['beta'])).all())

    def test_extract_permuted_false_inc_warmup(self):
        fit = self.fit
        ss = fit.extract(inc_warmup=True, permuted=False)
        num_samples = fit.sim['iter']
        self.assertEqual(ss.shape, (num_samples, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())

    def test_extract_thin(self):
        sm = self.sm
        fit = sm.sampling(chains=4, iter=2000, thin=2)

        # permuted True
        ss = fit.extract(permuted=True)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        self.assertEqual(sorted(ss.keys()), sorted({'alpha', 'beta', 'lp__'}))
        self.assertEqual(alpha.shape, (2000, 2, 3))
        self.assertEqual(beta.shape, (2000, 2))
        self.assertEqual(lp__.shape, (2000,))
        self.assertTrue((~np.isnan(alpha)).all())
        self.assertTrue((~np.isnan(beta)).all())
        self.assertTrue((~np.isnan(lp__)).all())

        # permuted False
        ss = fit.extract(permuted=False)
        self.assertEqual(ss.shape, (500, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())

        # permuted False inc_warmup True
        ss = fit.extract(inc_warmup=True, permuted=False)
        self.assertEqual(ss.shape, (1000, 4, 9))
        self.assertTrue((~np.isnan(ss)).all())

    def test_extract_dtype(self):
        dtypes = {"alpha": np.int, "beta": np.int}
        ss = self.fit.extract(dtypes = dtypes)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        self.assertEqual(alpha.dtype, np.dtype(np.int))
        self.assertEqual(beta.dtype, np.dtype(np.int))
        self.assertEqual(lp__.dtype, np.dtype(np.float))

    def test_extract_dtype_permuted_false(self):
        dtypes = {"alpha": np.int, "beta": np.int}
        pars = ['alpha', 'beta', 'lp__']
        ss = self.fit.extract(pars=pars, dtypes = dtypes, permuted=False)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        self.assertEqual(alpha.dtype, np.dtype(np.int))
        self.assertEqual(beta.dtype, np.dtype(np.int))
        self.assertEqual(lp__.dtype, np.dtype(np.float))

    def test_to_dataframe_permuted_true(self):
        ss = self.fit.extract(permuted=True)
        alpha = ss['alpha']
        beta = ss['beta']
        lp__ = ss['lp__']
        df = self.fit.to_dataframe(permuted=True)
        self.assertEqual(df.shape, (4000,7+9+6))
        for idx in range(2):
            for jdx in range(3):
                name = 'alpha[{},{}]'.format(idx+1,jdx+1)
                assert_array_equal(df[name].values,alpha[:,idx,jdx])
        for idx in range(2):
            name = 'beta[{}]'.format(idx+1)
            assert_array_equal(df[name].values,beta[:,idx])
        assert_array_equal(df['lp__'].values,lp__)
        # Test pars argument
        df = self.fit.to_dataframe(pars='alpha', permuted=True)
        self.assertEqual(df.shape, (4000,7+6+6))
        for idx in range(2):
            for jdx in range(3):
                name = 'alpha[{},{}]'.format(idx+1,jdx+1)
                assert_array_equal(df[name].values,alpha[:,idx,jdx])
        # Test pars and dtype argument
        df = self.fit.to_dataframe(pars='alpha',dtypes = {'alpha':np.int}, permuted=True)
        alpha_int = ss['alpha'].astype(np.int)
        self.assertEqual(df.shape, (4000,7+6+6))
        for idx in range(2):
            for jdx in range(3):
                name = 'alpha[{},{}]'.format(idx+1,jdx+1)
                assert_array_equal(df[name].values,alpha_int[:,idx,jdx])

    def test_to_dataframe_permuted_false_inc_warmup_false(self):
        fit = self.fit
        ss = fit.extract(permuted=False)
        df = fit.to_dataframe(permuted=False)
        num_samples = fit.sim['iter'] - fit.sim['warmup']
        num_chains = fit.sim['chains']
        self.assertEqual(df.shape, (num_samples*num_chains,3+9+6))
        alpha_index = 0
        for jdx in range(3):
            for idx in range(2):
                name = 'alpha[{},{}]'.format(idx+1,jdx+1)
                for n in range(num_chains):
                    assert_array_equal(
                    df.loc[df.chain == n, name].values,ss[:,n,alpha_index]
                    )
                alpha_index += 1
        for idx in range(2):
            name = 'beta[{}]'.format(idx+1)
            for n in range(num_chains):
                assert_array_equal(
                df.loc[df.chain == n, name].values,ss[:,n,6+idx]
                )
            for n in range(num_chains):
                assert_array_equal(df.loc[df.chain == n,'lp__'].values,ss[:,n,-1])
        diagnostic_type = {'divergent':int,'energy':float,'treedepth':int,
			                'accept_stat':float, 'stepsize':float, 'n_leapfrog':int}
        for n in range(num_chains):
            assert_array_equal(
                df.chain.values[n*num_samples:(n+1)*num_samples],
                n*np.ones(num_samples,dtype=np.int)
                )
            assert_array_equal(
                df.draw.values[n*num_samples:(n+1)*num_samples],
                np.arange(num_samples,dtype=np.int)
                )
            for diag, diag_type in diagnostic_type.items():
                assert_array_equal(
                df[diag+'__'].values[n*num_samples:(n+1)*num_samples],
                fit.get_sampler_params()[n][diag+'__'][-num_samples:].astype(diag_type)
                )

    def test_to_dataframe_permuted_false_inc_warmup_true(self):
        fit = self.fit
        ss = fit.extract(permuted=False, inc_warmup=True)
        df = fit.to_dataframe(permuted=False,inc_warmup=True)
        num_samples = fit.sim['iter']
        num_chains = fit.sim['chains']
        self.assertEqual(df.shape, (num_samples*num_chains,3+9+6))
        alpha_index = 0
        for jdx in range(3):
            for idx in range(2):
                name = 'alpha[{},{}]'.format(idx+1,jdx+1)
                for n in range(num_chains):
                    assert_array_equal(
                    df.loc[df.chain == n, name].values,ss[:,n,alpha_index]
                    )
                alpha_index += 1
        for idx in range(2):
            name = 'beta[{}]'.format(idx+1)
            for n in range(num_chains):
                assert_array_equal(
                df.loc[df.chain == n, name].values,ss[:,n,6+idx]
                )
            for n in range(num_chains):
                assert_array_equal(df.loc[df.chain == n,'lp__'].values,ss[:,n,-1])
                assert_array_equal(df.loc[
                n*fit.sim['n_save'][n]:n*fit.sim['n_save'][n]+fit.sim['warmup2'][n]-1, 'warmup'].values,
                np.ones(fit.sim['warmup2'][n]))
                assert_array_equal(df.loc[
                n*fit.sim['n_save'][n]+fit.sim['warmup2'][n]:
                (n+1)*fit.sim['n_save'][n]-1,'warmup'].values,
                np.zeros(fit.sim['warmup2'][n]))
        diagnostic_type = {'divergent':int,'energy':float,'treedepth':int,
			                'accept_stat':float, 'stepsize':float, 'n_leapfrog':int}
        for n in range(num_chains):
            assert_array_equal(
                df.chain.values[n*num_samples:(n+1)*num_samples],
                n*np.ones(num_samples,dtype=np.int)
                )
            assert_array_equal(
                df.draw.values[n*num_samples:(n+1)*num_samples],
                np.arange(num_samples,dtype=np.int)-int(fit.sim['warmup'])
                )
            for diag, diag_type in diagnostic_type.items():
                assert_array_equal(
                df[diag+'__'].values[n*num_samples:(n+1)*num_samples],
                fit.get_sampler_params()[n][diag+'__'][-num_samples:].astype(diag_type)
                )

    def test_to_dataframe_permuted_false_diagnostics_false(self):
        fit = self.fit
        ss = fit.extract(permuted=False)
        df = fit.to_dataframe(permuted=False,diagnostics=False)
        num_samples = fit.sim['iter'] - fit.sim['warmup']
        num_chains = fit.sim['chains']
        self.assertEqual(df.shape, (num_samples*num_chains,3+9))
        alpha_index = 0
        for jdx in range(3):
            for idx in range(2):
                name = 'alpha[{},{}]'.format(idx+1,jdx+1)
                for n in range(num_chains):
                    assert_array_equal(
                    df[name].loc[df.chain == n].values,ss[:,n,alpha_index]
                    )
                alpha_index += 1
        for idx in range(2):
            name = 'beta[{}]'.format(idx+1)
            for n in range(num_chains):
                assert_array_equal(
                df[name].loc[df.chain == n].values,ss[:,n,6+idx]
                )
            for n in range(num_chains):
                assert_array_equal(df.loc[df.chain == n,'lp__'].values,ss[:,n,-1])
            for n in range(num_chains):
                assert_array_equal(
                df.chain.values[n*num_samples:(n+1)*num_samples],
                n*np.ones(num_samples,dtype=np.int)
                )
                assert_array_equal(
                df.draw.values[n*num_samples:(n+1)*num_samples],
                np.arange(num_samples,dtype=np.int)
                )

    def test_to_dataframe_permuted_false_pars(self):
        fit = self.fit
        ss = fit.extract(permuted=False)
        df = fit.to_dataframe(permuted=False, pars='alpha')
        num_samples = fit.sim['iter'] - fit.sim['warmup']
        num_chains = fit.sim['chains']
        self.assertEqual(df.shape, (num_samples*num_chains,3+6+6))
        alpha_index = 0
        for jdx in range(3):
            for idx in range(2):
                name = 'alpha[{},{}]'.format(idx+1,jdx+1)
                for n in range(num_chains):
                    assert_array_equal(
                    df[name].loc[df.chain == n].values,ss[:,n,alpha_index]
                    )
                alpha_index += 1
        diagnostic_type = {'divergent':int,'energy':float,'treedepth':int,
			                'accept_stat':float, 'stepsize':float, 'n_leapfrog':int}
        for n in range(num_chains):
            assert_array_equal(
                df.chain.values[n*num_samples:(n+1)*num_samples],
                n*np.ones(num_samples,dtype=np.int)
                )
            assert_array_equal(
                df.draw.values[n*num_samples:(n+1)*num_samples],
                np.arange(num_samples,dtype=np.int)
                )
            for diag, diag_type in diagnostic_type.items():
                assert_array_equal(
                df[diag+'__'].values[n*num_samples:(n+1)*num_samples],
                fit.get_sampler_params()[n][diag+'__'][-num_samples:].astype(diag_type)
                )
