
.. _installation_cvodes:

.. currentmodule:: pystan

Installing PyStan with Support for Stiff ODE Solvers (CVODES)
=============================================================

Those using Stan functions which require the use of the SUNDIALS library (e.g.,
``integrate_ode_bdf``) should use the following instructions to install PyStan.

First, make sure that you have installed the following packages:

- Cython
- Numpy

Now install a version of PyStan which compiles and uses the SUNDIALS library:

::

    pip install https://github.com/stan-dev/pystan/archive/v2.18.0.0-cvodes.tar.gz


(Support for the SUNDIALS library is not included by default because it slows down compilation of every Stan program by several seconds.)

Consult the "Stan Language Manual" (linked to in the `Stan Documentation <http://mc-stan.org/users/documentation/index.html>`_) for an example of a complete Stan program with a system definition and solver call.
