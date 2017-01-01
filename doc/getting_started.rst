.. _getting-started:

.. currentmodule:: pystan

=================
 Getting started
=================

PyStan is the `Python <http://python.org>`_ interface for `Stan <http://mc-stan.org/>`_.

Prerequisites
=============

PyStan has the following dependencies:

- `Python <http://python.org>`_: 2.7, >=3.3
- `Cython <http://cython.org>`_: >=0.22
- `NumPy <http://numpy.org>`_: >=1.7

PyStan also requires that a C++ compiler be available to Python during
installation and at runtime. On Debian-based systems this is accomplished by
issuing the command ``apt-get install build-essential``.

Installation
============

.. note:: Installing PyStan involves compiling Stan. This may take
    a considerable amount of time.

Unix-based systems including Mac OS X
-------------------------------------

PyStan and the required packages may be installed from the `Python Package Index
<https://pypi.python.org/pypi>`_ using ``pip``.

::

   pip install pystan

To install PyStan from source, first make sure you have installed the
dependencies, then issue the commands:

::

   wget https://pypi.python.org/packages/source/p/pystan/pystan-2.14.0.0.tar.gz
   # alternatively, use curl, or a web browser
   tar zxvf pystan-2.14.0.0.tar.gz
   cd pystan-2.14.0.0
   python setup.py install
   cd ..  # change out of the source directory before importing pystan

Mac OS X users encountering installation problems may wish to consult the
`PyStan Wiki <https://github.com/stan-dev/pystan/wiki>`_ for possible solutions.

Windows
-------

PyStan on Windows requires Python 3.5 or higher and a working C++ compiler.  If
you have already installed Python 3.5 (or higher) and the Microsoft Visual C++
14.0 (or higher) compiler, running ``pip install pystan`` will install PyStan.
Note that you must specify ``n_jobs=1`` when drawing samples using Stan because
PyStan on Windows is not currently able to use multiple processors
simultaneously.

If you need to install a C++ compiler, you will find detailed installation
instructions in :ref:`windows`.

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
download the file :download:`8schools.stan <8schools.stan>` into
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

If `matplotlib <http://matplotlib.org/>`_ and `scipy <http://http://www.scipy.org/>`_ are installed, a visual summary may
also be displayed using the ``plot()`` method.

.. code-block:: python

    fit.plot()
