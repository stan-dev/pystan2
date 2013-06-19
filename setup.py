#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# This file is part of PyStan.
#
# PyStan is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# PyStan is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyStan.  If not, see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------------

LONG_DESCRIPTION    = \
"""
Python Interface to Stan, a package for Bayesian inference using
the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.
"""

NAME         = 'pystan'
DESCRIPTION  = 'Python interface to Stan, a package for Bayesian inference'
AUTHOR       = 'Allen B. Riddell',
AUTHOR_EMAIL = 'abr@ariddell.org',
URL          = 'https://github.com/ariddell/pystan'
LICENSE      = 'GPLv3'
CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Cython',
    'Development Status :: 2 - Pre-Alpha',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Information Analysis'
]
MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("pystan._api",
              ["pystan/_api.pyx"],
              language="c++",
              include_dirs=["pystan/stan/src/",
                            "pystan/stan/lib/boost_1.53.0/"],
              library_dirs=["pystan/bin"],
              libraries=["stanc"]
              )]

package_dir = {'pystan': 'pystan'}
package_data_pats = [
    '*.hpp',
    'io.pxd',
    'stan_fit.pxd',
    'stanc.pxd',
    'stanfit4model.pyx',
    'bin/*.a',
    'stan/src/stan/*.hpp',
    'stan/src/stan/gm/*.hpp',
    'stan/src/stan/io/*.hpp',
    'stan/src/stan/agrad/*.hpp',
    'stan/src/stan/mcmc/hmc/static/*.hpp',
    'stan/src/stan/mcmc/hmc/nuts/*.hpp',
    'stan/src/optimization/*.hpp',
    'stan/lib/boost_1.53.0/boost/date_time/posix_time/posix_time.hpp',
    'stan/lib/boost_1.53.0/boost/date_time/posix_time/posix_time_types.hpp',
    'stan/lib/boost_1.53.0/boost/math/special_functions/fpclassify.hpp',
    'stan/lib/boost_1.53.0/boost/random/additive_combine.hpp',
    'stan/lib/boost_1.53.0/boost/random/uniform_real_distribution.hpp',
]
package_data = {'pystan': package_data_pats}

setup(
    name=NAME,
    version=FULLVERSION,
    maintainer=AUTHOR,
    packages=['pystan'],
    ext_modules=cythonize(extensions),
    package_dir=package_dir,
    package_data=package_data,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
)
