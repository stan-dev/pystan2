# REF: rstan/example/test8schools.R
import os

import numpy as np

from pystan import StanModel, stan


def test8schools():

    model_name = "_8chools"
    sfile = os.path.join(os.path.dirname(__file__),
                         "../stan/src/models/misc/eight_schools/eight_schools.stan")
    m = StanModel(file=sfile, model_name=model_name, verbose=True)
    m.dso

    yam = StanModel(file=sfile, model_name=model_name, save_dso=False, verbose=True)
    yam.dso

    dat = dict(J=8, y=(28,  8, -3,  7, -1,  1, 18, 12),
               sigma=(15, 10, 16, 11,  9, 11, 10, 18))

    iter = 5020

    # HMC
    ss1 = m.sampling(data=dat, iter=iter, chains=4, algorithm='HMC', refresh=100)
    ss1son = stan(fit=ss1, data=dat, init_r=0.0001)
    ss1son = stan(fit=ss1, data=dat, init_r=0)
    ainfo1 = ss1.get_adaptation_info()
    lp1 = ss1.get_logposterior()
    yalp1 = ss1.get_logposterior(inc_warmup=False)
    sp1 = ss1.get_sampler_params()
    yasp1 = ss1.get_sampler_params(inc_warmup=False)
    gm1 = ss1.get_posterior_mean()
    print(gm1)

    # NUTS 1
    ss2 = m.sampling(data=dat, iter=iter, chains=4, refresh=100,
                     control=dict(metric="unit_e"))
    ainfo2 = ss2.get_adaptation_info()
    lp2 = ss2.get_logposterior()
    yalp2 = ss2.get_logposterior(inc_warmup=False)
    sp2 = ss2.get_sampler_params()
    yasp2 = ss2.get_sampler_params(inc_warmup=False)
    gm2 = ss2.get_posterior_mean()
    print(gm2)

    # NUTS 2
    ss3 = m.sampling(data=dat, iter=iter, chains=4, refresh=100)
    ainfo3 = ss3.get_adaptation_info()
    lp3 = ss3.get_logposterior()
    yalp3 = ss3.get_logposterior(inc_warmup=False)
    sp3 = ss3.get_sampler_params()
    yasp3 = ss3.get_sampler_params(inc_warmup=False)

    gm3 = ss3.get_posterior_mean()
    print(gm3)

    # Non-diag
    ss4 = m.sampling(data=dat, iter=iter, chains=4,
                     control=dict(metric='dense_e'), refresh=100)
    ainfo4 = ss4.get_adaptation_info()
    lp4 = ss4.get_logposterior()
    yalp4 = ss4.get_logposterior(inc_warmup=False)
    sp4 = ss4.get_sampler_params()
    yasp4 = ss4.get_sampler_params(inc_warmup=False)

    gm4 = ss4.get_posterior_mean()
    print(gm4)

    print(ss1)
    print(ss2)
    print(ss3)

    ss1.plot()
    ss1.traceplot()

    ss9 = m.sampling(data=dat, iter=iter, chains=4, refresh=10)

    iter = 52012

    ss = stan(sfile, data=dat, iter=iter, chains=4, sample_file='8schools.csv')

    print(ss)

    ss_inits = ss['inits']
    ss_same = stan(sfile, data=dat, iter=iter, chains=4,
                   seed=ss['stan_args'][0]['seed'], init=ss_inits,
                   sample_file='ya8schools.csv')

    b = np.allclose(ss['sim']['samples'], ss_same['sim']['samples'])
    # b is not true as ss is initialized randomly while ss.same is not.

    s = ss_same.summary(pars="mu", probs=(.3, .8))
    # not in python: print(ss.same, pars='theta', probs=c(.4, .8))
    print(ss.same)
