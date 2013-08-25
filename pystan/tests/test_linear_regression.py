import numpy as np
from pystan import stan

def test_linear_regression():
    np.random.seed(1)

    n = 10000
    p = 3

    beta_true = (1, 3, 5)
    X = np.random.normal(size=(n, p))
    X = (X - np.mean(X, axis=0)) / np.std(X, ddof=1, axis=0, keepdims=True)
    y = np.dot(X, beta_true) + np.random.normal(size=n)

    # OLS
    beta = np.linalg.lstsq(X, y)[0]

    model_code = """
    data {
        int<lower=0> N;
        int<lower=0> p;
        matrix[N,p] x;
        vector[N] y;
    }
    parameters {
        vector[p] beta;
        real<lower=0> sigma;
    }
    model {
        y ~ normal(x * beta, sigma);
    }
    """

    data = {'N': n, 'p': p, 'x': X, 'y': y}

    fit = stan(model_code=model_code, data=data, iter=2000)

    assert fit.sim['dims_oi'] == fit._get_param_dims()

    np.mean(fit.extract()['beta'], axis=0)
    np.mean(fit.extract()['sigma'])

    sigma = fit.extract()['sigma']
    beta = fit.extract()['beta']

    # mean of sigma is 1
    assert np.count_nonzero(np.abs(sigma - 1) < 0.05)
    assert all(np.abs(np.mean(beta, 0) - np.array(beta_true)) < 0.05)
