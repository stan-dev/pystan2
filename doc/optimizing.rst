
.. _optimizing:

.. currentmodule:: pystan

======================
 Optimization in Stan
======================

PyStan provides an interface to Stan's optimization methods. These methods
obtain a point estimate by maximizing the posterior function defined for
a model. The following example estimates the mean from samples assumed to be
drawn from normal distribution with known standard deviation:

.. math:

    y_1, \ldots, y_n \sim \text{normal}(\mu,1).

Specifying an improper prior for :math:`\mu` of :math:`p(\mu) \propto 1`,
the posterior obtains a maximum at the sample mean. The following Python
code illustrates how to use Stan's optimizer methods via a call to
``optimizing``:

.. code-block:: python

    import pystan
    import numpy as np

    ocode = """
    data {
        int<lower=1> N;
        real y[N];
    }
    parameters {
        real mu;
    }
    model {
        y ~ normal(mu, 1);
    }
    """
    sm = pystan.StanModel(model_code=ocode)
    y2 = np.random.normal(size=20)
    np.mean(y2)

    op = sm.optimizing(data=dict(y=y2, N=len(y2)))

    op
