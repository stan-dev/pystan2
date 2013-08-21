import numpy as np
from pystan import stan, StanModel

def test_matrix_param():
    model_code = """
    data {
    int<lower=2> K;
    int<lower=1> D;
    }
    parameters {
    matrix[K,D] beta;
    }
    model {
    for (k in 1:K)
        for (d in 1:D)
          beta[k,d] ~ normal(0,1);
    }"""
    fit = stan(model_code=model_code, data=dict(K=3,D=4))
    beta = fit.extract()['beta']
    assert beta.shape == (4000, 3, 4)
    assert np.mean(beta) < 4
    extracted = fit.extract(permuted=False)
    assert extracted.shape == (1000, 4, 13)
    assert np.mean(extracted[:,:,0:11]) < 4
    assert np.all(extracted[:,:,12] < 0)  # lp__


def test_matrix_param_order():
    model_code = """
    data {
    int<lower=2> K;
    }
    parameters {
    matrix[K,2] beta;
    }
    model {
    for (k in 1:K)
      beta[k,1] ~ normal(0,1);
    for (k in 1:K)
      beta[k,2] ~ normal(100,1);
    }"""
    fit = stan(model_code=model_code, data=dict(K=3))
    beta = fit.extract()['beta']
    assert beta.shape == (4000, 3, 2)
    beta_mean = np.mean(beta, axis=0)
    beta_colmeans = np.mean(beta_mean, axis=0)
    assert beta_colmeans[0] < 4
    assert beta_colmeans[1] > 100 - 4

def test_matrix_param_order_optimizing():
    model_code = """
    data {
    int<lower=2> K;
    }
    parameters {
    matrix[K,2] beta;
    }
    model {
    for (k in 1:K)
      beta[k,1] ~ normal(0,1);
    for (k in 1:K)
      beta[k,2] ~ normal(100,1);
    }"""
    sm = StanModel(model_code=model_code)
    op = sm.optimizing(data=dict(K=3))
    beta = op['par']['beta']
    assert beta.shape == (3, 2)
    beta_colmeans = np.mean(beta, axis=0)
    assert beta_colmeans[0] < 4
    assert beta_colmeans[1] > 100 - 4
