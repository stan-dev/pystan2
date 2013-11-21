# REF: rstan/example/test8schools2.R
import os

import numpy as np

from pystan import StanModel, stan


model_name = "_8chools"
schools_code = '''
  data {
    int<lower=0> J; // number of schools
    real y[J]; // estimated treatment effects
    real<lower=0> sigma[J]; // s.e. of effect estimates
  }
  parameters {
    real mu;
    real<lower=0> tau;
    real eta[J];
  }
  transformed parameters {
    real theta[J];
    for (j in 1:J)
      theta[j] <- mu + tau * eta[j];
  }
  model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
  }
'''
m = StanModel(model_code=schools_code, model_name=model_name, verbose=True)

J = 8
y = (28,  8, -3,  7, -1,  1, 18, 12)
sigma = (15, 10, 16, 11,  9, 11, 10, 18)

iter = 1000
dat = dict(J=J, y=y, sigma=sigma)
ss1 = m.sampling(data=dat, iter=iter, chains=4, refresh=100)

print(ss1)
ss1.traceplot()

ss = stan(model_code=schools_code, data=dat, iter=iter, chains=4,
          sample_file='8schools.csv')
print(ss)
ss.plot()


# using previous fitted objects
ss2 = stan(fit=ss, data=dat, iter=2000)
ss2.summary(probs=[0.38])
ss2.summary(probs=[0.48])
# ss2.summary(probs=[0.48], use_cache=False)

ss3 = stan(fit=ss, data=dat, save_dso=False) # save_dso taks no effect
yss = stan(model_code=schools_code, data=dat, iter=iter, chains=4,
           sample_file='8schools.csv', save_dso=False)
#save.image()

# print(ss, use_cache=False)
# ls(ss@.MISC)
# print(ss)
# ls(ss@.MISC)

print("stan with init=0")
ss4 = stan(fit=ss, data=dat, init=0)


def initfun(chain_id=0):
    return dict(mu=np.random.random(),
                theta=np.random.random(size=J),
                tau=np.random.exponential(chain_id + 1))

try:
    ss5 = stan(fit=ss, data=dat, init=initfun)
except RuntimeError as e:
    assert 'eta missing' in str(e)

#mode = ss.get_cppo_mode()
ss.get_stancode()
#rstan:::is_sf_valid(ss)

## print the dso
#ss@stanmodel@dso

samples = ss.extract()

# do.call(cat, get_adaptation_info(ss))
ss8 = stan(fit=ss, data=dat, test_grad=True)
#print(ss8)
#ss8.summary()
#ss8.plot()
#ss8.traceplot()
#ss.get_adaptation_info()
##get_cppo_mode(ss8)
print(ss8.get_inits())

print(ss8.get_seed())
ss8.get_stancode()
sm8 = ss8.get_stanmodel()
#ss8.extract()

#m8 = ss8.get_posterior_mean()
#print(m8)
#print(ss8)

#
ss9 = stan(fit=ss, data=dat,  seed=-1, chains=1)
# seed is too big, so in config_argss, it will be turned to NA
ss10 = stan(fit=ss, data=dat, seed=4294967295, chains=1, iter=1000)
ss10.get_seed()
ss11 = stan(fit=ss, data=dat, seed="4294967295", chains=1, iter=1000)

m11 = ss11.get_posterior_mean()
print(m11)
print(ss11)
