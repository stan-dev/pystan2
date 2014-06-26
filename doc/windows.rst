.. _windows:

.. currentmodule:: pystan

===================
 PyStan on Windows
===================

Caveats
-------

- PyStan on Windows does not support multiprocessing. You *must* specify
  ``n_jobs=1`` when calling the ``stan`` function or using the ``sampling``
  method.

Quickstart
----------

Those using (or are willing to use) the Anaconda Python distribution and a free
compiler provided by Microsoft will find installation of PyStan relatively
painless. Instructions about downloading and configuring the compiler, Visual
Studio 2008 Express, may be found on the Menpo wiki: `Setting up a development
environment
<https://github.com/menpo/menpo/wiki/%5BDevelopment-Windows%5D-Setting-up-a-development-environment>`_.

Once you have Visual Studio 2008 installed and Anaconda's Python 2.7, you can
install a binary version of PyStan provided by Patrick Snape with the following
command:

::

    conda install -c patricksnape pystan


Detailed instructions
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
