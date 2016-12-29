#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2013-2015, PyStan developers
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
import ast
import codecs
import os
import platform
import sys

LONG_DESCRIPTION = open('README.rst').read()
NAME         = 'pystan'
DESCRIPTION  = 'Python interface to Stan, a package for Bayesian inference'
AUTHOR       = 'PyStan Developers'
AUTHOR_EMAIL = 'stan-users@googlegroups.com'
URL          = 'https://github.com/stan-dev/pystan'
LICENSE      = 'GPLv3'
CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Cython',
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Information Analysis'
]


# VersionFinder from from django-compressor
class VersionFinder(ast.NodeVisitor):
    def __init__(self):
        self.version = None

    def visit_Assign(self, node):
        if node.targets[0].id == '__version__':
            self.version = node.value.s


def read(*parts):
    filename = os.path.join(os.path.dirname(__file__), *parts)
    with codecs.open(filename, encoding='utf-8') as fp:
        return fp.read()


def find_version(*parts):
    finder = VersionFinder()
    finder.visit(ast.parse(read(*parts)))
    return finder.version


###############################################################################
# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function

# For some commands, use setuptools
if len(set(('develop', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist_wininst', 'install_egg_info', 'build_sphinx',
           'egg_info', 'easy_install', 'upload', 'bdist_wheel',
           '--single-version-externally-managed',
            )).intersection(sys.argv)) > 0:
    import setuptools
    extra_setuptools_args = dict(
        install_requires=['Cython>=0.22,!=0.25.1', 'numpy >= 1.7'],
        zip_safe=False, # the package can run out of an .egg file
        include_package_data=True,
    )
else:
    extra_setuptools_args = dict()

###############################################################################

from distutils.errors import CCompilerError, DistutilsError
from distutils.extension import Extension

stan_include_dirs = ['pystan/stan/src',
                     'pystan/stan/lib/stan_math_2.14.0/',
                     'pystan/stan/lib/stan_math_2.14.0/lib/eigen_3.2.9',
                     'pystan/stan/lib/stan_math_2.14.0/lib/boost_1.62.0',
                     'pystan/stan/lib/stan_math_2.14.0/lib/cvodes_2.9.0/include']
stan_macros = [
    ('BOOST_RESULT_OF_USE_TR1', None),
    ('BOOST_NO_DECLTYPE', None),
    ('BOOST_DISABLE_ASSERTS', None),
    ('FUSION_MAX_VECTOR_SIZE', 12),  # for parser, stan-dev/pystan#222
]
extra_compile_args = [
    '-Os',
    '-ftemplate-depth-256',
    '-Wno-unused-function',
    '-Wno-uninitialized',
]

if platform.platform().startswith('Win'):
    extra_compile_args = [
        '/EHsc',
        '-DBOOST_DATE_TIME_NO_LIB',
    ]


stanc_sources = [
    "pystan/stan/src/stan/lang/ast_def.cpp",
    "pystan/stan/src/stan/lang/grammars/bare_type_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/expression07_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/expression_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/functions_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/indexes_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/program_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/semantic_actions_def.cpp",
    "pystan/stan/src/stan/lang/grammars/statement_2_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/statement_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/term_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/var_decls_grammar_inst.cpp",
    "pystan/stan/src/stan/lang/grammars/whitespace_grammar_inst.cpp",
]

extensions = [
    Extension("pystan._api",
              ["pystan/_api.pyx"] + stanc_sources,
              language='c++',
              define_macros=stan_macros,
              include_dirs=stan_include_dirs,
              extra_compile_args=extra_compile_args),
    Extension("pystan._chains",
              ["pystan/_chains.pyx"],
              language='c++',
              define_macros=stan_macros,
              include_dirs=stan_include_dirs,
              extra_compile_args=extra_compile_args),
    # _misc.pyx does not use Stan libs
    Extension("pystan._misc", ["pystan/_misc.pyx"], language='c++')
]


## package data
package_data_pats = ['*.hpp', '*.pxd', '*.pyx', 'tests/data/*.csv']

# get every file under pystan/stan/src and pystan/stan/lib
stan_files_all = sum(
    [[os.path.join(path.replace('pystan/', ''), fn) for fn in files]
     for path, dirs, files in os.walk('pystan/stan/src/')], [])

stan_math_files_all = sum(
    [[os.path.join(path.replace('pystan/', ''), fn) for fn in files]
     for path, dirs, files in os.walk('pystan/math/')], [])

lib_files_all = sum(
    [[os.path.join(path.replace('pystan/', ''), fn) for fn in files]
     for path, dirs, files in os.walk('pystan/stan/lib/')], [])

package_data_pats += stan_files_all
package_data_pats += stan_math_files_all
package_data_pats += lib_files_all


def setup_package():
    metadata = dict(name=NAME,
                    version=find_version("pystan", "__init__.py"),
                    maintainer=AUTHOR,
                    maintainer_email=AUTHOR_EMAIL,
                    packages=['pystan',
                              'pystan.tests',
                              'pystan.external',
                              'pystan.external.pymc',
                              'pystan.external.enum',
                              'pystan.external.scipy'],
                    ext_modules=extensions,
                    package_data={'pystan': package_data_pats},
                    platforms='any',
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    long_description=LONG_DESCRIPTION,
                    classifiers=CLASSIFIERS,
                    **extra_setuptools_args)
    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1]
                               in ('--help-commands', 'egg_info', '--version', 'clean')):
        # For these actions, neither Numpy nor Cython is required.
        #
        # They are required to succeed when pip is used to install PyStan
        # when, for example, Numpy is not yet present.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        dist = setup(**metadata)
    else:
        import distutils.core
        distutils.core._setup_stop_after = 'commandline'
        from distutils.core import setup
        try:
            from Cython.Build import cythonize
            # FIXME: if header only works, no need for numpy.distutils at all
            from numpy.distutils.command import install
        except ImportError:
            raise SystemExit("Cython>=0.22 and NumPy are required.")

        metadata['ext_modules'] = cythonize(extensions)
        dist = setup(**metadata)

        metadata['cmdclass'] = {'install': install.install}
    try:
        dist.run_commands()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
    except (IOError, os.error) as exc:
        from distutils.util import grok_environment_error
        error = grok_environment_error(exc)
    except (DistutilsError, CCompilerError) as msg:
            raise SystemExit("error: " + str(msg))


if __name__ == '__main__':
    setup_package()
