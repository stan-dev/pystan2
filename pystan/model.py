#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

try:
    # Python 3
    from collections.abc import Callable, Sequence
except ImportError:
    # Python 2.7
    from collections import Callable, Sequence
import datetime
import importlib
import imp
import logging
from numbers import Number
import os
import random
import tempfile
import string
import sys

try:
    import hashlib
except ImportError:
    import md5 as hashlib

from distutils.core import Extension

import Cython
from Cython.Build.Inline import _get_build_extension
from Cython.Build.Dependencies import cythonize

import numpy as np

import pystan.api
from pystan.constants import MAX_UINT
import pystan.misc

def load_module(module_name, module_path):
    """Load the module named `module_name` from  `module_path`
    independently of the Python version."""
    if hasattr(importlib, 'find_loader'):
        # Python 3
        loader = importlib.find_loader(module_name, [module_path])
        return loader.load_module()
    else:
        # Python 2.7
        module_info = imp.find_module(module_name, [module_path])
        return imp.load_module(module_name, *module_info)


# NOTE: StanModel instance stores references to a compiled, uninstantiated
# C++ model.
class StanModel:
    """
    Model described in Stan's modeling language compiled from C++ code.

    Instances of StanModel are typically created indirectly by the functions
    `stan` and `stanc`.

    Parameters
    ----------
    file : string {'filename', 'file'}
        If filename, the string passed as an argument is expected to
        be a filename containing the Stan model specification.

        If file, the object passed must have a 'read' method (file-like
        object) that is called to fetch the Stan model specification.

    charset : string, 'utf-8' by default
        If bytes or files are provided, this charset is used to decode.

    model_name: string, 'anon_model' by default
        A string naming the model. If none is provided 'anon_model' is
        the default. However, if `file` is a filename, then the filename
        will be used to provide a name.

    model_code : string
        A string containing the Stan model specification. Alternatively,
        the model may be provided with the parameter `file`.

    stanc_ret : dict
        A dict returned from a previous call to `stanc` which can be
        used to specify the model instead of using the parameter `file` or
        `model_code`.

    boost_lib : string
        The path to a version of the Boost C++ library to use instead of
        the one supplied with PyStan.

    eigen_lib : string
        The path to a version of the Eigen C++ library to use instead of
        the one in the supplied with PyStan.

    save_dso : boolean, True by default
        Indicates whether the dynamic shared object (DSO) compiled from
        C++ code will be saved for use in a future Python session.

    verbose : boolean, False by default
        Indicates whether intermediate output should be piped to the console.
        This output may be useful for debugging.

    kwargs : keyword arguments
        Additional arguments passed to `stanc`.

    Attributes
    ----------
    model_name : string
    model_code : string
        Stan code for the model.
    model_cpp : string
        C++ code for the model.
    dso : builtins.module
        Python module created by compiling the C++ code for the model.

    Methods
    -------
    show
        Print the Stan model specification.
    sampling
        Draw samples from the model.
    optimizing
        Obtain a point estimate by maximizing the log-posterior.
    get_cppcode
        Return the C++ code for the module.
    get_cxxflags
        Return the 'CXXFLAGS' used for compiling the model.

    See also
    --------
    stanc: Compile a Stan model specification
    stan: Fit a model using Stan

    Notes
    -----
    Instances of StanModel can be saved for use across Python sessions only
    if `save_dso` is set to True during the construction of StanModel objects.

    Even if `save_dso` is True, models cannot be loaded on platforms that
    differ from the one on which the model was compiled.

    More details of Stan, including the full user's guide and
    reference manual can be found at <URL: http://mc-stan.org/>.

    There are three ways to specify the model's code for `stan_model`.

    1. parameter `model_code`, containing a string to whose value is
       the Stan model specification,

    2. parameter `file`, indicating a file (or a connection) from
       which to read the Stan model specification, or

    3. parameter `stanc_ret`, indicating the re-use of a model
         generated in a previous call to `stanc`.

    References
    ----------

    The Stan Development Team (2013) _Stan Modeling Language User's
    Guide and Reference Manual_.  <URL: http://mc-stan.org/>.

    Examples
    --------
    >>> model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
    >>> model_code; m = StanModel(model_code=model_code)
    ... # doctest: +ELLIPSIS
    'parameters ...
    >>> m.model_name
    'anon_model'

    """
    def __init__(self, file=None, charset='utf-8', model_name="anon_model",
                 model_code=None, stanc_ret=None, boost_lib=None,
                 eigen_lib=None, save_dso=True, verbose=False, **kwargs):

        if stanc_ret is None:
            stanc_ret = pystan.api.stanc(file=file,
                                         model_code=model_code,
                                         model_name=model_name,
                                         verbose=verbose, **kwargs)
        if not isinstance(stanc_ret, dict):
            raise ValueError("stanc_ret must be an object returned by stanc.")
        stanc_ret_keys = {'status', 'model_code', 'model_cppname',
                          'cppcode', 'model_name'}
        if not all(n in stanc_ret_keys for n in stanc_ret):
            raise ValueError("stanc_ret lacks one or more of the keys: "
                             "{}".format(str(stanc_ret_keys)))
        elif stanc_ret['status'] != 0:  # success == 0
            raise ValueError("stanc_ret is not a successfully returned "
                             "dictionary from stanc.")
        self.model_cppname = stanc_ret['model_cppname']
        self.model_name = stanc_ret['model_name']
        self.model_code = model_code
        self.model_cppcode = stanc_ret['cppcode']

        msg = "COMPILING THE C++ CODE FOR MODEL {} NOW."
        logging.warn(msg.format(self.model_name))
        if verbose:
            msg = "OS: {}, Python: {}, Cython {}".format(sys.platform,
                                                         sys.version,
                                                         Cython.__version__)
            logging.info(msg)
        if boost_lib is not None:
            # FIXME: allow boost_lib, eigen_lib to be specified
            raise NotImplementedError
        if eigen_lib is not None:
            raise NotImplementedError

        key = tuple([self.model_code, self.model_cppcode, sys.version_info,
                     sys.executable])
        module_name = ("stanfit4" + self.model_name + '_' +
                       hashlib.md5(str(key).encode('utf-8')).hexdigest())

        temp_dir = tempfile.mkdtemp()
        lib_dir = os.path.join(temp_dir, 'pystan')
        pystan_dir = os.path.dirname(__file__)
        include_dirs = [lib_dir,
                        pystan_dir,
                        os.path.join(pystan_dir, "stan/src/"),
                        os.path.join(pystan_dir, "stan/lib/eigen_3.1.3"),
                        os.path.join(pystan_dir, "stan/lib/boost_1.53.0/")]
        library_dirs = [os.path.join(pystan_dir, "bin")]
        libraries = ['stan']

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        model_cpp_file = os.path.join(lib_dir, self.model_cppname + '.hpp')
        with open(model_cpp_file, 'w') as outfile:
            outfile.write(self.model_cppcode)

        pyx_file = os.path.join(lib_dir, module_name + '.pyx')
        pyx_template_file = os.path.join(pystan_dir, 'stanfit4model.pyx')
        with open(pyx_template_file) as infile:
            s = infile.read()
            template = string.Template(s)
        with open(pyx_file, 'w') as outfile:
            s = template.safe_substitute(model_cppname=self.model_cppname)
            outfile.write(s)

        extension = Extension(name=module_name,
                              language="c++",
                              sources=[pyx_file],
                              include_dirs=include_dirs,
                              library_dirs=library_dirs,
                              libraries=libraries,
                              extra_compile_args=['-O3'])

        cython_include_dirs = ['.', pystan_dir]
        build_extension = _get_build_extension()
        build_extension.extensions = cythonize([extension],
                                            include_path=cython_include_dirs,
                                            quiet=not verbose)
        build_extension.build_temp = os.path.dirname(pyx_file)
        build_extension.build_lib = lib_dir

        if not verbose:
            # silence stderr for compilation
            orig_stderr = pystan.misc.redirect_stderr()

        build_extension.run()

        if not verbose:
            # restore stderr
            os.dup2(orig_stderr, sys.stderr.fileno())

        module_path = lib_dir
        self.module = load_module(module_name, module_path)
        self.fit_class = getattr(self.module, "StanFit4" + self.model_cppname)

    def show(self):
        return self.model_code

    @property
    def dso(self):
        return self.module

    def get_cppcode(self):
        return self.model_cppcode

    def get_cxxflags(self):
        # FIXME: implement this?
        raise NotImplementedError

    def optimizing(self, data=None, seed=random.randint(0, MAX_UINT),
                   init='random', check_data=True, sample_file=None,
                   verbose=False):
        """Obtain a point estimate by maximizing the joint posterior.

        Parameters
        ----------

        """
        if data is None:
            data = {}
        data_r, data_i = pystan.misc._split_data(data)
        fit = self.fit_class(data_r, data_i)
        m_pars = fit.get_param_names()
        idx_of_lp = m_pars.index('lp__')
        del m_pars[idx_of_lp]
        if isinstance(init, Number):
            init = str(init)
        elif isinstance(init, Callable) or isinstance(init, Iterable) or \
                isinstance(init, str):
            pass
        else:
            raise ValueError("Incorrect specification of initial values.")
        seed = int(seed)

        stan_args = {'init': init, 'seed': seed, 'point_estimate': True}

        stan_args.update({'test_grad': False})  # not to test gradient
        stan_args = pystan.misc._get_valid_stan_args(stan_args)
        ret = fit._call_sampler(stan_args)
        optim = None
        return optim

    def sampling(self, data=None, pars=None, chains=4, iter=2000,
                 warmup=None, thin=1, seed=random.randint(0, MAX_UINT),
                 init='random', check_data=True, sample_file=None,
                 verbose=False, **kwargs):
        """
        NB: remove this check_data business -- either you supply a dict
        or you dont!

        """
        # NOTE: in this function, iter masks iter() the python function.
        # If this ever turns out to be a problem just add:
        # iter_ = iter
        # del iter  # now builtins.iter is available
        if data is None:
            data = {}
        if warmup is None:
            warmup = int(iter // 2)

        # FIXME: remove this check? check_data stuff in rstan isn't pythonic
        if check_data:
            if not isinstance(data, dict):
                raise ValueError("data must be a dictionary.")

        data_r, data_i = pystan.misc._split_data(data)
        fit = self.fit_class(data_r, data_i)
        # store a copy of the data passed to fit in the class
        fit.data = {}
        fit.data.update(data_i)
        fit.data.update(data_r)

        m_pars = fit._get_param_names()
        p_dims = fit._get_param_dims()
        if pars is not None and len(pars) > 0:
            if not all(p in m_pars for p in pars):
                pars = np.asarray(pars)
                unmatched = pars[np.invert(np.in1d(pars, m_pars))]
                msg = "No parameter(s): {}; sampling not done."
                raise ValueError(msg.format(', '.join(pars[unmatched])))

        if chains < 1:
            raise ValueError("The number of chains is less than one; sampling"
                             "not done.")

        args_list = pystan.misc._config_argss(chains=chains, iter=iter,
                                              warmup=warmup, thin=thin,
                                              init=init, seed=seed,
                                              sample_file=sample_file,
                                              **kwargs)

        # number of samples saved after thinning
        warmup2 = 1 + (warmup - 1) // thin
        n_kept = 1 + (iter - warmup - 1) // thin
        n_save = n_kept + warmup2

        samples, rets = [], []  # samples and return values
        if kwargs.get('test_grad') is None:
            mode = "SAMPLING"
        else:
            mode = "TESTING GRADIENT"
        # FIXME: use concurrent.futures to parallelize this
        for i in range(chains):
            if kwargs.get('refresh') is None or kwargs.get('refresh') > 0:
                chain_num = i + 1
                msg = "{} FOR MODEL {} NOW (CHAIN {})."
                logging.info(msg.format(mode, self.model_name, chain_num))
            ret, samples_i = fit._call_sampler(args_list[i])
            samples.append(samples_i)
            # call_sampler in stan_fit.hpp will raise a std::runtime_error
            # if the return value is non-zero. Cython will generate a
            # RuntimeError from this.
            # FIXME: should one mimic rstan and "return" an empty StanFit?
            # That is, should I wipe fit's attributes and return that?

        inits_used = pystan.misc._organize_inits([s['inits'] for s in samples],
                                                 m_pars, p_dims)

        # test_gradient mode: don't sample
        if samples[0]['test_grad']:
            fit.sim = {'num_failed': [s['num_failed'] for s in samples]}
            return fit

        perm_lst = [np.random.permutation(n_kept) for _ in range(chains)]
        fnames_oi = fit._get_param_fnames_oi()
        n_flatnames = len(fnames_oi)
        fit.sim = {'samples': samples,
                   'iter': iter,
                   'warmup': warmup,
                   'n_save': [n_save] * chains,
                   'warmup2': [warmup2] * chains,
                   'permutation': perm_lst,
                   'pars_oi': fit._get_param_names_oi(),
                   'dims_oi': fit._get_param_dims(),
                   'fnames_oi': fnames_oi,
                   'n_flatnames': n_flatnames}
        fit.model_name = self.model_name
        fit.model_pars = m_pars
        fit.par_dims = p_dims
        fit.mode = 0
        fit.inits = inits_used
        fit.stan_args = args_list
        fit.stanmodel = self
        fit.date = datetime.datetime.now().isoformat()
        return fit
