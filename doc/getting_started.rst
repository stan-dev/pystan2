.. _getting_started:

.. currentmodule:: pystan

=================
 Getting Started
=================

Introduction
============

PyStan is the `Python <http://python.org>`_ interface for Stan.

Prerequisites
=============

PyStan has the following dependencies:

- `Python <http://python.org>`_: 3.3 or greater
- `Cython <http://cython.org>`_: 0.19.1 or greater
- `NumPy <http://numpy.org>`_: 1.6.2 or greater

PyStan also requires that a C++ compiler be available to Python during
installation and at runtime. On Debian-based systems this is accomplished by
issuing the command `apt-get install build-essential`.

Installation
============

If you have `pip <https://pypi.python.org/pypi/pip>`_ installed, PyStan can be
installed with a single command::

    pip install -e https://github.com/ariddell/pystan.git

Alternatively, download the package and install manually.

Using PyStan
============

The module's name is `pystan` so we load the module as follows:

.. code-block:: python

    import pystan

Example 1: Eight Schools
------------------------

This is an example in Section 5.5 of Gelman et al (2003), which studied coaching
effects from eight schools. For simplicity, we call this example "eight
schools."

.. code-block:: python

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

In this model, we let `theta` be transformed parameters of `mu` and `eta`
instead of directly declaring theta as parameters. By parameterizing this way,
the sampler will run more efficiently.

.. FIXME: fill in rest when features added

The object `fit`, returned from function `stan` stores samples from the
posterior distribution. The method `extract` extracts samples into a dictionary
of arrays for parameters of interest, or just an array.

.. code-block:: python

    la = fit.extract(permuted=True)  # return a dictionary of arrays
    mu = la['mu']

    ## return an array of three dimensions: iterations, chains, parameters
    a = fit.extract(permuted=False)

.. FIXME: fill in rest when features added

.. code-block:: python

    eta = fit.extract(permuted=True)['eta']
    np.mean(eta, axis=0)

    array([ 0.38781197,  0.02531919, -0.14462554,  0.00364737, -0.32423842,
        -0.2012991 ,  0.4047828 ,  0.12202298])
