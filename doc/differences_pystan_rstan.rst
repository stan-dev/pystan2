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

Saving objects
==============

PyStan uses ``pickle`` to save objects for future use.

Python:

.. code-block:: python

    import pickle
    import pystan

    model = pystan.StanModel(model_code=model_code)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # load it at some future point

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

R:

.. code-block:: r

    library(rstan)

    model = stan_model(model_code=model_code)
    save(model, file='model.rdata')

