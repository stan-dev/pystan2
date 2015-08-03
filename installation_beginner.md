# Detailed Installation Instructions

Installing PyStan requires installing:

- Python
- Python dependencies
- PyStan

# Prerequisite knowledge

It is highly recommended to know what `bash` is and the basics of
navigating the terminal.  You can review or learn it from the
[Software Carpentry](http://software-carpentry.org/) bash lesson here:
[http://swcarpentry.github.io/shell-novice/](http://swcarpentry.github.io/shell-novice/).
Lessons 1 - 3 are proabably the most important.

## Installing Python

The easiest way to install Python is to use the
[https://store.continuum.io/cshop/anaconda/](Anaconda) distribution of
python.  It can be downloaded here:
[http://continuum.io/downloads#34](http://continuum.io/downloads#34).

This is because PyStan (and many python tools) require packages (aka
modules) that have C dependencies.  These types of dependencies are
unable to be installed (at least easily) using `pip`, which is a
common way to install python packages.  Anaconda ships with it's own
package manager (that also plays nicely with `pip`) called `conda`,
and comes with many of the data analytics packages and dependencies
pre-installed.

Don't worry about Anaconda ruining your current Python installation,
can can be easily uninstalled (described below).

### Windows

The anaconda installer should be able to be double-clicked and
installed.  Use all of the defaults for installation except make sure
to check Make Anaconda the default Python.

### Macs

After downloading the installer, double click the .pkg file and follow
the instructions on the screen.  Use all of the defaults for
installation.

### Linux

After downloading the installer run: `bash
Anaconda-2.3.0-Linux-x86_64.sh` in the directory where you installed

### Uninstalling Anaconda

The default location for anaconda can be found in your home directory.
Typically this means it in in the `~/anaconda` or `~/anaconda3`
directory when you open a terminal.

## Python dependencies

If you used the Anaconda installer, numpy and cython should already be
installed, so additional dependencies should not be needed.  However,
should you need to install additional dependencies, we can use `conda`
to install them as such:

- open a terminal
- type `conda install numpy` to install numpy or replace `numpy` with
the package you need to install


## Installing PyStan

Since we have the `numpy` and `cython` dependencies we need, we can
install the latest version of PyStan using `pip`.  To do so:

- Open a terminal
- type `pip install pystan`
