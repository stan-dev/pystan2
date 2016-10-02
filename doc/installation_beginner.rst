
.. _installation_beginner:

.. currentmodule:: pystan

Detailed Installation Instructions
==================================

The following is addressed to an audience who is just getting started with
Python and would benefit from additional guidance on how to install PyStan.

Installing PyStan requires installing:

-  Python
-  Python dependencies
-  PyStan

Prerequisite knowledge
======================

It is highly recommended to know what ``bash`` is and the basics of
navigating the terminal. You can review or learn it from the `Software
Carpentry <http://software-carpentry.org/>`__ bash lesson here:
http://swcarpentry.github.io/shell-novice/.

Lessons 1 - 3 are proabably the most important.

Installing Python
-----------------

The easiest way to install Python is to use the Anaconda distribution
of python. It can be downloaded here: http://continuum.io/downloads.

This is because PyStan (and many python tools) require packages (aka
modules) that have C dependencies. These types of dependencies are
unable to be installed (at least easily) using ``pip``, which is a
common way to install python packages. Anaconda ships with it's own
package manager (that also plays nicely with ``pip``) called ``conda``,
and comes with many of the data analytics packages and dependencies
pre-installed.

Don't worry about Anaconda ruining your current Python installation, can
can be easily uninstalled (described below).

Anaconda is not a requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda is not an absolute requirement to get ``pystan`` to work. As
long as you can get the necessary python dependencies installed, pystan
will work. If you want to install Anaconda, follow the Windows, Macs,
and Linux instructions below.

Windows
~~~~~~~

The Anaconda installer should be able to be double-clicked and
installed. Use all of the defaults for installation except make sure to
check Make Anaconda the default Python.

Macs
~~~~

After downloading the installer, double click the .pkg file and follow
the instructions on the screen. Use all of the defaults for
installation.

Linux
~~~~~

After downloading the installer execute the associated shell script. For
example, if the file downloaded were named ``Anaconda3-4.1.1-Linux-x86_64.sh``
you would enter ``bash Anaconda3-4.1.1-Linux-x86_64.sh`` in the directory where
you downloaded the file.

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

Installing PyStan
-----------------

Since we have the ``numpy`` and ``cython`` dependencies we need, we can
install the latest version of PyStan using ``pip``. To do so:

-  Open a terminal
-  type ``pip install pystan``
