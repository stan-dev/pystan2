import numpy as np
from pystan import stan

def test_array_param():
    """
    Make sure shapes are getting unraveled correctly. Mixing up row-major and
    column-major data is a potential issue.
    """
    model_code = """
    data {
      int<lower=2> K;
    }
    parameters {
      real beta[K,1,2];
    }
    model {
      for (k in 1:K)
        beta[k,1,1] ~ normal(0,1);
      for (k in 1:K)
        beta[k,1,2] ~ normal(100,1);
    }"""
    fit = stan(model_code=model_code, data=dict(K=4))

    # extract, permuted
    beta = fit.extract()['beta']
    assert beta.shape == (4000, 4, 1, 2)
    beta_mean = np.mean(beta, axis=0)
    assert beta_mean.shape == (4, 1, 2)
    assert np.all(beta_mean[:, 0, 0] < 4)
    assert np.all(beta_mean[:, 0, 1] > 100 - 4)

    # extract, permuted=False
    extracted = fit.extract(permuted=False)
    assert extracted.shape == (1000, 4, 9)
    # in theory 0:4 should be
    # 'beta[0,0,0]'
    # 'beta[1,0,0]'
    # 'beta[2,0,0]'
    # 'beta[3,0,0]'
    #
    # and 4:8 should be
    # 'beta[0,0,1]'
    # 'beta[1,0,1]'
    # 'beta[2,0,1]'
    # 'beta[3,0,1]'
    assert np.all(np.mean(extracted[:, :, 0:4], axis=(0, 1)) < 4)
    assert np.all(np.mean(extracted[:, :, 4:8], axis=(0, 1)) > 100 - 4)
    assert np.all(extracted[:, :, 8] < 0)  # lp__

    # optimizing
    sm = fit.stanmodel
    op = sm.optimizing(data=dict(K=4))
    beta = op['par']['beta']
    assert beta.shape == (4, 1, 2)
    assert np.all(beta[:, 0, 0] < 4)
    assert np.all(beta[:, 0, 1] > 100 - 4)
