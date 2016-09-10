.. _windows:

.. currentmodule:: pystan

===================
 PyStan on Windows
===================

PyStan is supported under Windows with the following caveats:

- Python 3.5 or higher must be used.
- When drawing samples ``n_jobs=1`` must be used. (PyStan on Windows cannot use multiple processors in parallel.)

PyStan requires a working C++ compiler. Configuring such a compiler is
typically the most challenging step in getting PyStan running.

Installing Python
=================

There several ways of installing PyStan on Windows. The following instructions
assume you have installed Python 3.5 (or higher) as packaged in the `Anaconda
Python distribution <https://www.continuum.io/downloads#windows>`_. (Make sure
you install the Python 3 variant of Anaconda.) The Anaconda distribution is
well-maintained and includes packages such as Numpy which PyStan requires. The
following instructions assume that you are using Windows 7.  (`Windows 10
disregards user choice and user privacy
<https://www.eff.org/deeplinks/2016/08/windows-10-microsoft-blatantly-disregards-user-choice-and-privacy-deep-dive>`_.)

Installing a C++ Compiler
=========================

This section describes how to install the Microsoft Visual C++ 14.0 compiler.
It is likely that PyStan will work with more recent Microsoft Visual C++
compilers as well.

Navigate to the `Visual C++ Build Tools
<http://landinghub.visualstudio.com/visual-cpp-build-tools>`_ page and click on
"Download Visual C++ Build Tools 2015".

If you encounter problems you may find the `Windows Compilers
<https://wiki.python.org/moin/WindowsCompilers>`_ page on the Python Wiki
useful. Note that on Windows 7 you may need to update the installed version of
the Microsoft .NET Framework before installing the Visual C++ Build Tools.

Installing PyStan
=================

Once you have the compiler installed, installing PyStan is easy. Open the
application called "Command Prompt" (``cmd.exe``) and enter the following
command::

    pip install pystan

You can verify that everything was installed successfully by opening up the
Python terminal (run ``python`` from a command prompt) and drawing samples from
a very simple model::

    >>> import pystan
    >>> model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
    >>> model = pystan.StanModel(model_code=model_code)
    >>> y = model.sampling(n_jobs=1).extract()['y']
    >>> y.mean()  # with luck the result will be near 0

Again, remember that using ``n_jobs=1`` when calling ``sampling`` is required
as PyStan on Windows does not support sampling using multiple processors in
parallel.
