.. _windows:

.. currentmodule:: pystan

===================
 PyStan on Windows
===================

Caveats
-------

- PyStan on Windows does not support multiprocessing. You must specify
  ``n_jobs=1`` when calling the ``stan`` function or the ``sampling`` method.

Quickstart
----------

If you have Visual Studio 2008 installed and are using Python 2.7 through
Anacaonda, you can use a binary version of PyStan provided by Patrick Snape.
The following ``conda`` command will install PyStan 2.2.0.0:

::

    conda install -c patricksnape pystan

Detailed Instructions
---------------------

When you provide your model code to (Py)Stan, Stan generates and compiles C++
code that corresponds to your model. In order to compile the generated C++ code,
Stan needs a C++ compiler. Because this compiled Stan model communicates with
Python, the compiler used *should be the same compiler that compiled Python*.
The following instructions assume that you are using Python 2.7.x and that your
version of Python has been compiled with Visual Studio 2008.

In order to compile on Windows you will need to make modifications to
``setup.py`` and ``pystan/model.py``. Two flags required for Visual Studio 2008
are:

    - ``/Ehsc`` which turns on exceptions for boost
    - ``-DBOOST_DATE_TIME_NO_LIB`` which solves a bug in linking boost

These flags need to be added in ``setup.py`` and in ``model.py``.

Stan also needs to be patched for Visual Studio 2008. These patches may be
incorporated into Stan in future versions. Consult `issue #632
<https://github.com/stan-dev/stan/issues/632>`_ for details on these patches. 
