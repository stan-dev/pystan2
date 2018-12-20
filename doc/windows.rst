.. _windows:

.. currentmodule:: pystan

===================
 PyStan on Windows
===================

PyStan is partially supported under Windows with the following caveats:

- Python 2.7: Doesn't support parallel sampling. When drawing samples ``n_jobs=1`` must be used)
- Python 3.5 or higher: Parallel sampling is supported
- MSVC compiler is not supported.

PyStan requires a working C++ compiler. Configuring such a compiler is
typically the most challenging step in getting PyStan running.

PyStan is tested against the mingw-w64 compiler which works on both Python versions (2.7, 3.x)
and supports x86 and x64.

Due to problems with MSVC template deduction, functions with Eigen library are failing.
Until this and other bugs are fixed no support is provided for Windows + MSVC.
Currently, no fix is known for this problem, other than to change the compiler to GCC or clang-cl.


Installing Python
=================

There several ways of installing PyStan on Windows. The following instructions
assume you have installed Python as packaged in the `Anaconda
Python distribution <https://www.anaconda.com/download/#windows>`_
or `Miniconda distribution <https://conda.io/miniconda.html>`_.
The Anaconda distribution is well-maintained and includes packages such as Numpy
which PyStan requires. The following instructions assume that you are using Windows 7 
(`Windows 10 disregards user choice and user privacy
<https://www.eff.org/deeplinks/2016/08/windows-10-microsoft-blatantly-disregards-user-choice-and-privacy-deep-dive>`_).

Open Command prompt
===================

All the following commands are written to a command line.
You can open the command line with

- open "Anaconda prompt"
- open "Command prompt" ``cmd.exe`` (if conda is found on your PATH).

Test conda package manager by::

    conda info

To update conda package manager to the latest version::

    conda update conda

Create a conda virtual environment (optional)
=============================================

It is a good practice to keep specific projects on their on conda virtual environments
to prevent unnecessary package collisions. Create a new conda environment with::

    conda create -n stan_env python=3.7

where ``stan_env`` is the name of the environment.

After this activate environment with::

    conda activate stan_env

or if your conda doesn't include ``conda activate`` use::

    activate stan_env

To close the environment type::

    deactivate

Installing C++ compiler
=======================

There are several ways to install mingw-w64 compiler toolchain, but in these instructions
install compiler with ``conda`` package manager which comes with the Anaconda package.

To install mingw-w64 compiler type::

    conda install libpython m2w64-toolchain -c msys2

This will install

- ``libpython`` package which is needed to import mingw-w64. <https://anaconda.org/anaconda/libpython>
- mingw-w64 toolchain. <https://anaconda.org/msys2/m2w64-toolchain>

``libpython`` setups automatically ``distutils.cfg`` file, but if that is failed
use the following instructions to setup it manually

In `PYTHONPATH\\Lib\\distutils` create `distutils.cfg` with text editor (e.g. `notepad`, `notepad++`) and add the following lines::

    [build]
    compiler=mingw32

To find the correct `distutils` path, run `python`::

    >>> import distutils
    >>> print(distutils.__file__)


Install dependencies
====================

It is recommended that on Windows the dependencies are installed with conda and ``conda-forge`` channel.
Required dependencies are ``numpy`` and ``cython``.::

    conda install numpy cython -c conda-forge

Optional dependencies are ``matplotlib``, ``scipy`` and ``pandas``.::

    conda install matplotlib scipy pandas -c conda-forge

Installing PyStan
=================

You can install PyStan with either pip (recommended) or conda

with pip::

    pip install pystan

And with conda

    conda install pystan -c conda-forge


You can verify that everything was installed successfully by opening up the
Python terminal (run ``python`` from a command prompt) and drawing samples from
a very simple model::

    >>> import pystan
    >>> model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
    >>> model = pystan.StanModel(model_code=model_code)
    >>> y = model.sampling().extract()['y']
    >>> y.mean()  # with luck the result will be near 0


Steps
=====

With pip

    conda install numpy cython matplotlib scipy pandas -c conda-forge
    pip install pystan``

With conda

    conda install numpy cython matplotlib scipy pandas pystan -c conda-forge
