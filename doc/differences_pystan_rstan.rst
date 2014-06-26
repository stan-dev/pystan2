.. _differences-pystan-rstan:

.. currentmodule:: pystan

======================================
 Differences between PyStan and RStan
======================================

While PyStan attempts to maintain API compatibility with RStan, there are
certain unavoidable differences between Python and R.

Methods and attributes
======================

Methods are invoked in different ways: ``fit.summary()`` and ``fit.extract()``
(Python) vs. ``summary(fit)`` and ``extract(fit)`` (R).

Attributes are accessed in a different manner as well: ``fit.sim`` (Python) vs.
``fit@sim`` (R).

Dictionaries instead of Lists
=============================

Where RStan uses lists, PyStan uses (ordered) dictionaries.

Python:

.. code-block:: python

    fit.extract()['theta']

R:

.. code-block:: r

    extract(fit)$theta

Reusing models and saving objects
=================================

PyStan uses ``pickle`` to save objects for future use.

Python:

.. code-block:: python

    import pickle
    import pystan

    # bernoulli model
    model_code = """
        data {
          int<lower=0> N;
          int<lower=0,upper=1> y[N];
        }
        parameters {
          real<lower=0,upper=1> theta;
        }
        model {
          for (n in 1:N)
              y[n] ~ bernoulli(theta);
        }
        """
    data = dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    model = pystan.StanModel(model_code=model_code)
    fit = model.sampling(data=data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # load it at some future point

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # run with different data
    fit = model.sampling(data=dict(N=5, y=[1, 1, 0, 1, 0]))

R:

.. code-block:: r

    library(rstan)

    model = stan_model(model_code=model_code)
    save(model, file='model.rdata')

See also :ref:`avoiding-recompilation`.
