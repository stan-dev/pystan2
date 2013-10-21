.. _getting-started:

.. currentmodule:: pystan

=================
 Getting Started
=================

Introduction
============

PyStan is the `Python <http://python.org>`_ interface for Stan.

.. beta software notice, remove this eventually

PyStan aims to reproduce the functionality present in RStan. There are a number
of features present in RStan that have yet to be implemented in PyStan.  If you
find a feature missing that you use frequently please `file a bug report
<https://github.com/ariddell/pystan/issues>`_ so developers can better direct
their efforts.

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

Unix-based Systems including Mac OS X
-------------------------------------

PyStan and the required packages may be installed from the `Python Package Index
<https://pypi.python.org/pypi>`_ using ``pip``. (A recent version of
``setuptools`` is required under Python 2.7; installation from source may prove
easier.)

::

   pip install numpy Cython
   pip install pystan

To install PyStan from source, first make sure you have installed the
dependencies, then issue the commands:

::

   wget https://github.com/ariddell/pystan/archive/0.2.2.zip
   unzip 0.2.2.zip
   cd pystan-0.2.2
   python setup.py install
   cd ..  # change out of the source directory before using pystan

Windows
-------

Installation under Windows it not well tested. `Python(x,y)
<https://code.google.com/p/pythonxy/>`_ provides a distribution of Python that
includes NumPy, Cython, and the GNU Compiler `MinGW
<http://docs.cython.org/src/tutorial/appendix.html>`_. (Note that Cython should
be selected during installation of Python(x,y), as it is not installed by
default).

.. note:: Installing PyStan involves compiling Stan. This may take
    a considerable amount of time.

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

If `matplotlib <http://matplotlib.org/>`_ is installed, a visual summary may
also be displayed using the ``plot()`` method.

.. code-block:: python

    fit.plot()
