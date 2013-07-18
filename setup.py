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

import os
import distutils.core
from distutils.errors import CCompilerError, DistutilsError
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy.distutils.command import install, install_clib
from numpy.distutils.misc_util import InstallableLib

## static libraries
stan_include_dirs = ["pystan/stan/src/",
                     "pystan/stan/lib/eigen_3.1.3/",
                     "pystan/stan/lib/boost_1.53.0/"]

stan_macros = [('BOOST_RESULT_OF_USE_TR1', None),
               ('BOOST_NO_DECLTYPE', None),
               ('BOOST_DISABLE_ASSERTS', None)]

libstan_sources = [
    "pystan/stan/src/stan/agrad/rev/var_stack.cpp",
    "pystan/stan/src/stan/math/matrix.cpp",
    "pystan/stan/src/stan/agrad/matrix.cpp"
]

libstan = ('stan', {'sources': libstan_sources,
                    'include_dirs': stan_include_dirs,
                    'extra_compile_args': ['-O3'],
                    'macros': stan_macros})

## extensions
stanc_sources = [
    "pystan/stan/src/stan/gm/grammars/var_decls_grammar_inst.cpp",
    "pystan/stan/src/stan/gm/grammars/expression_grammar_inst.cpp",
    "pystan/stan/src/stan/gm/grammars/statement_2_grammar_inst.cpp",
    "pystan/stan/src/stan/gm/grammars/statement_grammar_inst.cpp",
    "pystan/stan/src/stan/gm/grammars/term_grammar_inst.cpp",
    "pystan/stan/src/stan/gm/grammars/program_grammar_inst.cpp",
    "pystan/stan/src/stan/gm/grammars/whitespace_grammar_inst.cpp",
    "pystan/stan/src/stan/gm/ast_def.cpp"
]

extensions = [Extension("pystan._api",
                        ["pystan/_api.pyx"] + stanc_sources,
                        define_macros=stan_macros,
                        include_dirs=stan_include_dirs,
                        extra_compile_args=['-O3'])]

## package data
package_data_pats = ['*.hpp', '*.pxd', '*.pyx']

# get every file under pystan/stan/src and pystan/stan/lib
stan_files_all = sum(
    [[os.path.join(path.replace('pystan/', ''), fn) for fn in files]
     for path, dirs, files in os.walk('pystan/stan/src/')], [])

lib_files_all = sum(
    [[os.path.join(path.replace('pystan/', ''), fn) for fn in files]
     for path, dirs, files in os.walk('pystan/stan/lib/')], [])

package_data_pats += stan_files_all
package_data_pats += lib_files_all

## setup
if __name__ == '__main__':
    distutils.core._setup_stop_after = 'commandline'
    dist = setup(
        name=NAME,
        version=FULLVERSION,
        maintainer=AUTHOR,
        packages=['pystan', 'pystan.tests'],
        ext_modules=cythonize(extensions),
        libraries=[libstan],
        package_dir={'pystan': 'pystan'},
        package_data={'pystan': package_data_pats},
        maintainer_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        # use numpy.distutils machinery to install libstan.a
        cmdclass={'install': install.install,
                  'install_clib': install_clib.install_clib},
    )
    # use numpy.distutils machinery to install libstan.a
    dist.installed_libraries = [InstallableLib(libstan[0],
                                               libstan[1],
                                               'pystan/bin/')]
    try:
        dist.run_commands()
    except KeyboardInterrupt:
        raise SystemExit("interrupted")
    except (IOError, os.error) as exc:
        from distutils.util import grok_environment_error
        error = grok_environment_error(exc)
    except (DistutilsError,
            CCompilerError) as msg:
            raise SystemExit("error: " + str(msg))
