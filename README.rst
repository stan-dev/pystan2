PyStan: The Python Interface to Stan
====================================

.. image:: https://raw.githubusercontent.com/stan-dev/logos/master/logo.png
    :alt: Stan logo
    :scale: 50 %

|pypi| |travis| |appveyor|

**PyStan** provides a Python interface to Stan, a package for Bayesian inference
using the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.

For more information on `Stan <http://mc-stan.org>`_ and its modeling language,
see the Stan User's Guide and Reference Manual at `http://mc-stan.org/
<http://mc-stan.org/>`_.

Important links
---------------

- HTML documentation: https://pystan.readthedocs.org
- Issue tracker: https://github.com/stan-dev/pystan/issues
- Source code repository: https://github.com/stan-dev/pystan
- Stan: http://mc-stan.org/
- Stan User's Guide and Reference Manual (pdf) available at http://mc-stan.org

Related projects
----------------

- Scikit-learn integration: `pystan-sklearn <https://github.com/rgerkin/pystan-sklearn>`_ by @rgerkin.

Similar projects
----------------

- PyMC: http://pymc-devs.github.io/pymc/

Detailed Installation Instructions
----------------------------------
Detailed installation instructions can be found in the
`doc/installation_beginner.md <https://github.com/chendaniely/pystan/blob/develop/doc/installation_beginner.rst/>`_ file.

Quick Installation
------------------

`NumPy  <http://www.numpy.org/>`_ and `Cython <http://www.cython.org/>`_
(version 0.22 or greater) are required. `matplotlib <http://matplotlib.org/>`_
is optional.

PyStan and the required packages may be installed from the `Python Package Index
<https://pypi.python.org/pypi>`_ using ``pip``.

::

   pip install pystan

Alternatively, if Cython (version 0.22 or greater) and NumPy are already
available, PyStan may be installed from source with the following commands

::

   git clone --recursive https://github.com/stan-dev/pystan.git
   cd pystan
   python setup.py install

If you encounter an ``ImportError`` after compiling from source, try changing
out of the source directory before attempting ``import pystan``. On Linux and
OS X ``cd /tmp`` will work.

Example
-------

::

    import pystan
    import numpy as np
    import matplotlib.pyplot as plt

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
            theta[j] = mu + tau * eta[j];
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
    plt.show() 

.. |pypi| image:: https://badge.fury.io/py/pystan.png
    :target: https://badge.fury.io/py/pystan
    :alt: pypi version

.. |travis| image:: https://travis-ci.org/stan-dev/pystan.png?branch=master
    :target: https://travis-ci.org/stan-dev/pystan
    :alt: travis-ci build status

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/stan-dev/pystan?branch=master
    :target: https://ci.appveyor.com/project/ariddell/pystan/branch/master
    :alt: appveyor-ci build status
