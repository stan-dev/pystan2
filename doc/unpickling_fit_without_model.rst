.. _unpickling_fit_without_model:

.. currentmodule:: pystan

=====================================================
 Unpickling fit without Model instance (experimental)
=====================================================

This feature is experimental and nothing is guaranteed.

In PyStan it is needed to import model instance before unpickling the fit object.
This is commonly done by pickling model instance and fit object together in an object
that keeps the order (``list``, ``tuple``, ``OrderedDict``, ``dict`` with python 3.6+).

Sometimes mistakes are done, and fit objects are pickled without the model instance,
and there is no way to reload the model from other pickled files. Rebuilding the model
from Stan-code does not work in this case, because the module name contains random parts
that are decided at the runtime.
To still get data and samples saved to fit object, PyStan has ``unpickle_fit`` -function
in its experimental submodule. The function will create a fake ``StanModel`` instance
with the correct module name, which enables one to unpickle fit object without the
original model instance. This means, that the model object should not be used to
anything which includes the functionality wrapped in the fit object.

The suggested use of this feature is to enable users to extract samples with
``.extract()`` method (and access other information stored to ``fit.sim`` object).
After extraction it is recommended to save samples in another format, such as
``dict`` , netCDF4 (``arviz.InferenceData``) or tabular format (``pandas.DataFrame``).

Example
=======

Following example will show how to unpickle fit object without model instance and
save the resulting samples in other formats.

.. code-block:: python

    import pystan
    import pystan.experimental

    path = "path/to/fit.pickle"
    fit = pystan.experimental.unpickle_fit(path)

Example on how to save extracted samples to with pickle

.. code-block:: python

    import pickle
    with open("path/to/fit_dict.pickle", "wb") as f:
        pickle.dump(fit.extract(), protocol=pickle.HIGHEST_PROTOCOL)

Example on how to save extracted samples + other information with ArviZ to netCDF4.

.. code-block:: python

    import arviz as az
    # for more advanced function use, see https://arviz-devs.github.io/arviz/generated/arviz.from_pystan.html
    idata = az.from_pystan(fit)
    idata.to_netcdf("path/to/fit.nc")

Example on how to save extracted samples to csv with pandas

.. code-block:: python

    # using pystan.misc.to_dataframe -function works also with the older fit objects
    df = pystan.misc.to_dataframe(fit)
    df.to_csv("path/to/fit.csv", index=False)
