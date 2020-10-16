
.. _installation_beginner:

.. currentmodule:: pystan

Detailed Installation Instructions
==================================

The following is addressed to an audience who is just getting started with
Python and would benefit from additional guidance on how to install PyStan.

Installing PyStan requires installing:

-  Python
-  C++ -compiler
-  Python dependencies
-  PyStan

Prerequisite knowledge
======================

It is highly recommended to know what ``bash`` is and the basics of
navigating the terminal. You can review or learn it from the `Software
Carpentry <http://software-carpentry.org/>`_ (`bash lesson here
<http://swcarpentry.github.io/shell-novice/>`_).

Lessons 1 - 3 are probably the most important.

Installing Python
-----------------

The easiest way to install Python is to use the Anaconda distribution
of python. It can be downloaded `here <https://www.anaconda.com/download/>`_.

This is because PyStan (and many python tools) require packages (aka
modules) that have C dependencies. These types of dependencies are
unable to be installed (at least easily) using ``pip``, which is a
common way to install python packages. Anaconda ships with it's own
package manager (that also plays nicely with ``pip``) called ``conda``,
and comes with many of the data analytics packages and dependencies
pre-installed.

Don't worry about Anaconda ruining your current Python installation, it
can be easily uninstalled (described below).

Anaconda is not a requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda is not an absolute requirement to get ``pystan`` to work. As
long as you can get the necessary python dependencies installed, pystan
will work. If you want to install Anaconda, follow the Windows, Macs,
and Linux instructions below.

Linux
~~~~~

After downloading the installer execute the associated shell script. For
example, if the file downloaded were named ``Anaconda3-4.1.1-Linux-x86_64.sh``
you would enter ``bash Anaconda3-4.1.1-Linux-x86_64.sh`` in the directory where
you downloaded the file.

Macs
~~~~

After downloading the installer, double click the .pkg file and follow
the instructions on the screen. Use all of the defaults for
installation.

Windows
~~~~~~~

PyStan on Windows is *partially* supported. See `PyStan on Windows <_windows>`_.

The Anaconda installer should be able to be double-clicked and
installed. Use all of the defaults for installation except make sure to
check Make Anaconda the default Python.

Uninstalling Anaconda
~~~~~~~~~~~~~~~~~~~~~

The default location for anaconda can be found in your home directory.
Typically this means it in in the ``~/anaconda`` or ``~/anaconda3``
directory when you open a terminal.

Python dependencies
-------------------

If you used the Anaconda installer, numpy and cython should already be
installed, so additional dependencies should not be needed. However,
should you need to install additional dependencies, we can use ``conda``
to install them as such:

-  open a terminal
-  type ``conda install numpy`` to install numpy or replace ``numpy``
   with the package you need to install
-  common additional packages include ``arviz``, ``matplotlib``, ``pandas``,
   ``scipy``

Setting up C++ compiler
-----------------------

PyStan 2.19+ needs a C++14 compatible compiler. For GCC 4.9.3+ and
GCC 5+ versions are up-to-date. To update your compiler, follow general
instructions given for each platform.

-  Linux: Install compiler with ``apt``/``yum``
-  Macs: Install latest XCode, or use ``brew`` or ``macports``
-  Windows: Follow instructions in `PyStan on Windows <_windows>`_.

To use ``gcc``/``clang`` compiler from the ``conda`` follow instructions in
https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html
and set-up your ``CC`` and ``CXX`` environmental variables as given below. Instructions below
assume default Anaconda environment. For ``conda-forge`` follow installation
instructions in anaconda.org.

-  open a terminal
-  Linux: ``conda install gcc_linux-64 gxx_linux-64 -c anaconda``
-  Macs: ``conda install clang_osx-64 clangxx_osx-64 -c anaconda``

To check your compiler version

- Open a terminal
- Linux: type ``gcc --version``
- OSX: type ``clang --version``

To use specific C++ compiler (Linux: ``gcc_linux-64``, ``gxx_linux-64``, OSX: ``clang_osx-64``, ``clangxx_osx-64``), either update your ``CC`` and ``CXX`` environmental
variables or set-up your path to include folder containing compilers
(e.g.``which gcc_linux-64``)

- type ``export CC=gcc_linux-64`` and ``export CXX=g++_linux-64``
- type ``export PATH=/path/to/Anaconda/bin:$PATH``

Conda will create a new name for the compiler. You'll need to search in the `<path to conda>/bin` folder to find the name. `which python` will show this location. For example:

- Open a terminal
- OSX: type ``ls <path to python env>/bin/ | grep clang``

You should see two compilers like ``x86_64-apple-darwin13.4.0-clang`` and ``x86_64-apple-darwin13.4.0-clang++``.

- type ``export CC=x86_64-apple-darwin13.4.0-clang`` and ``export CXX=x86_64-apple-darwin13.4.0-clang++``



Installing PyStan
-----------------

Since we have the ``numpy`` and ``cython`` dependencies we need, we can
install the latest version of PyStan using ``pip``. To do so:

-  open a terminal
-  type ``pip install pystan``


Checking that it works
----------------------

Open a python prompt, and run the following code

.. code-block:: python

    import pystan
    model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
    model = pystan.StanModel(model_code=model_code, verbose=True) 

If it fails with "fatal error: 'Python.h' file not found", please use conda or install the appropriate python headers library.

If it fails with "fatal error: 'ios' file not found", please run `xcode-select --install`, or open XCode and agree to the licensing terms.

If it fails with a different error, please file a bug.
