PyStan: The Python Interface to Stan
====================================

.. image:: https://travis-ci.org/stan-dev/pystan.png
        :target: https://travis-ci.org/stan-dev/pystan

PyStan has an interface similar to that of RStan. For an introduction to Stan
and RStan see `http://mc-stan.org/ <http://mc-stan.org/>`_ and `RStan Getting
Started <https://code.google.com/p/stan/wiki/RStanGettingStarted>`_.

.. beta notice, remove eventually

PyStan aims to reproduce the functionality present in RStan. There are a few
features present in RStan that have yet to be implemented in PyStan.  If you
find a feature missing that you use frequently please `file a bug report
<https://github.com/stan-dev/pystan/issues>`_ so developers can better direct
their efforts.

Important links
---------------

- Source code repo: https://github.com/stan-dev/pystan
- HTML documentation: https://pystan.readthedocs.org
- Issue tracker: https://github.com/stan-dev/pystan/issues
- Stan: http://mc-stan.org/

Similar projects
----------------

- PyMC: http://pymc-devs.github.io/pymc/

Installation
------------

`NumPy  <http://www.numpy.org/>`_ and `Cython <http://www.cython.org/>`_
(version 0.19.1 or greater) are required. `matplotlib <http://matplotlib.org/>`_
is optional.

PyStan and the required packages may be installed from the `Python Package Index
<https://pypi.python.org/pypi>`_ using ``pip``.

::

   pip install numpy Cython
   pip install pystan

Alternatively, if Cython (version 0.19 or greater) and NumPy are already
available, PyStan may be installed from source with the following commands

::

   git clone https://github.com/stan-dev/pystan.git
   cd pystan
   python setup.py install

If you encounter an ``ImportError`` after compiling from source, try changing
out of the source directory before attempting ``import pystan``. For example, on
Linux and OS X ``cd /tmp`` would work.

Example
-------

::

    import pystan
    import numpy as np

    schools_code = """
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
    """

    schools_dat = {'J': 8,
                   'y': [28,  8, -3,  7, -1,  1, 18, 12],
                   'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

    fit = pystan.stan(model_code=schools_code, data=schools_dat,
                      iter=1000, chains=4)

    print(fit)

    eta = fit.extract(permuted=True)['eta']
    np.mean(eta, axis=0)

    # if matplotlib is installed (optional, not required), a visual summary and
    # traceplot are available
    fit.plot()

