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

- `Python <http://python.org>`_: 2.7 or 3.3
- `Cython <http://cython.org>`_: 0.19.1 or greater
- `NumPy <http://numpy.org>`_: 1.6.2 or greater

PyStan also requires that a C++ compiler be available to Python during
installation and at runtime. On Debian-based systems this is accomplished by
issuing the command ``apt-get install build-essential``.

Installation
============

PyStan currently needs to be installed from source. First make sure you have
installed the dependencies, then issue the commands:

::

   git clone https://github.com/ariddell/pystan.git
   cd pystan
   python setup.py install

Using PyStan
============

The module's name is ``pystan`` so we load the module as follows:

.. code-block:: python

    import pystan

Example 1: Eight Schools
------------------------

The "eight schools" example appears in Section 5.5 of Gelman et al. (2003),
which studied coaching effects from eight schools.

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

In this model, we let ``theta`` be transformed parameters of ``mu`` and ``eta``
instead of directly declaring theta as parameters. By parameterizing this way,
the sampler will run more efficiently.

In PyStan, we can also specify the Stan model using a file. For example, we can
download the file `8schools.stan
<http://wiki.stan.googlecode.com/git/rstangettingstarted/8schools.stan>`_ into
our working directory and use the following call to ``stan`` instead:

.. code-block:: python

    fit1 = pystan.stan(file='8schools.stan', data=schools_dat, iter=1000, chains=4)

Once a model is fitted, we can use the fitted result as an input to the model
with other data or settings. This saves us time compiling the C++ code for the
model. By specifying the parameter ``fit`` for the ``stan`` function, we can fit
the model again. For example, if we want to sample more iterations, we proceed
as follows:

.. code-block:: python

    fit2 = pystan.stan(fit=fit1, data=schools_dat, iter=10000, chains=4)

The object ``fit``, returned from function ``stan`` stores samples from the
posterior distribution. The ``fit`` object has a number of methods, including
``plot`` and ``extract``. We can also print the ``fit`` object and receive
a summary of the posterior samples as well as the log-posterior (which has the
name ``lp__``).

The method ``extract`` extracts samples into a dictionary of arrays for
parameters of interest, or just an array.

.. code-block:: python

    la = fit.extract(permuted=True)  # return a dictionary of arrays
    mu = la['mu']

    ## return an array of three dimensions: iterations, chains, parameters
    a = fit.extract(permuted=False)

.. code-block:: python

    print(fit)
