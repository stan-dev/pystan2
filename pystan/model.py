#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

from pystan._compat import PY2, string_types, implements_to_string
from collections import OrderedDict
if PY2:
    from collections import Callable, Iterable
    import md5 as hashlib
else:
    from collections.abc import Callable, Iterable
    import hashlib
import datetime
import importlib
import imp
import io
import logging
from numbers import Number
import os
import random
import tempfile
import string
import sys

from distutils.core import Extension

import Cython
from Cython.Build.Inline import _get_build_extension
from Cython.Build.Dependencies import cythonize

import numpy as np

import pystan.api
from pystan.constants import MAX_UINT
import pystan.misc

logger = logging.getLogger('pystan')
logger.setLevel(logging.INFO)


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
@implements_to_string
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
        self.save_dso = save_dso

        msg = "COMPILING THE C++ CODE FOR MODEL {} NOW."
        logger.warning(msg.format(self.model_name))
        if verbose:
            msg = "OS: {}, Python: {}, Cython {}".format(sys.platform,
                                                         sys.version,
                                                         Cython.__version__)
            logger.info(msg)
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
                        os.path.join(pystan_dir, "stan/src"),
                        os.path.join(pystan_dir, "stan/lib/eigen_3.2.0"),
                        os.path.join(pystan_dir, "stan/lib/boost_1.54.0")]
        library_dirs = [os.path.join(pystan_dir, "bin")]
        libraries = ['stan']

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        model_cpp_file = os.path.join(lib_dir, self.model_cppname + '.hpp')
        with io.open(model_cpp_file, 'w') as outfile:
            outfile.write(self.model_cppcode)

        pyx_file = os.path.join(lib_dir, module_name + '.pyx')
        pyx_template_file = os.path.join(pystan_dir, 'stanfit4model.pyx')
        with io.open(pyx_template_file, 'r', encoding='utf-8') as infile:
            s = infile.read()
            template = string.Template(s)
        with io.open(pyx_file, 'w') as outfile:
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

        redirect_stderr = not verbose and pystan.misc._has_fileno(sys.stderr)
        if redirect_stderr:
            # silence stderr for compilation
            orig_stderr = pystan.misc._redirect_stderr()

        try:
            build_extension.run()
        finally:
            if redirect_stderr:
                # restore stderr
                os.dup2(orig_stderr, sys.stderr.fileno())

        self.module = load_module(module_name, lib_dir)
        self.fit_class = getattr(self.module, "StanFit4" + self.model_cppname)

    def __str__(self):
        # NOTE: returns unicode even for Python 2.7, implements_to_string
        # decorator creates __unicode__ and __str__
        s = u"StanModel object '{}' coded as follows:\n{}"
        return s.format(self.model_name, self.model_code)

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

    def __getstate__(self):
        """Specify how instances are to be pickled
        self.module is unpicklable, for example.
        """
        state = self.__dict__.copy()
        if state['save_dso']:
            module_filename = state['module'].__file__
            state['module_filename'] = module_filename
            state['module_name'] = state['module'].__name__
            with io.open(module_filename, 'rb') as f:
                state['module_bytes'] = f.read()
        del state['module']
        del state['fit_class']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.save_dso:
            # the following attributes are temporary and exist only in the
            # pickled object to facilitate reloading the module
            module_filename = self.module_filename
            module_bytes = self.module_bytes
            module_name = self.module_name
            del self.module_filename
            del self.module_bytes
            del self.module_name
            temp_dir = tempfile.mkdtemp()
            lib_dir = os.path.join(temp_dir, 'pystan')
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            module_basename = os.path.basename(module_filename)
            with io.open(os.path.join(lib_dir, module_basename), 'wb') as f:
                f.write(module_bytes)
            try:
                self.module = load_module(module_name, lib_dir)
                self.fit_class = getattr(self.module, "StanFit4" + self.model_cppname)
            except Exception as e:
                logger.warning(e)
                logger.warning("Something went wrong while unpickling "
                               "the StanModel. Consider recompiling.")

    def optimizing(self, data=None, seed=None,
                   init='random', sample_file=None, method="Newton",
                   verbose=False, **kwargs):
        """Obtain a point estimate by maximizing the joint posterior.

        Parameters
        ----------
        data : dict
            A Python dictionary providing the data for the model. Variables
            for Stan are stored in the dictionary as expected. Variable
            names are the keys and the values are their associated values.
            Stan only accepts certain kinds of values; see Notes.

        seed : int, optional
            The seed, a positive integer for random number generation. Only
            one seed is needed when multiple chains are used, as the other
            chain's seeds are generated from the first chain's to prevent
            dependency among random number streams. By default, seed is
            ``random.randint(0, MAX_UINT)``.

        init : {0, '0', 'random', function returning dict, list of dict}, optional
            Specifies how initial parameter values are chosen: 0 or '0'
            initializes all to be zero on the unconstrained support; 'random'
            generates random initial values; list of size equal to the number
            of chains (`chains`), where the list contains a dict with initial
            parameter values; function returning a dict with initial parameter
            values. The function may take an optional argument `chain_id`.

        sample_file : string, optional
            File name specifying where samples for *all* parameters and other
            saved quantities will be written. If not provided, no samples
            will be written. If the folder given is not writable, a temporary
            directory will be used. When there are multiple chains, an
            underscore and chain number are appended to the file name.
            By default do not write samples to file.

        method : {"BFGS", "Nesterov", "Newton"}, optional
            Name of optimization method to be used. Default is Newton's method.

        verbose : boolean, optional
            Indicates whether intermediate output should be piped to the console.
            This output may be useful for debugging. False by default.

        Returns
        -------
        optim : dict or None
            Dictionary with components 'par' and 'value' if the optimization is
            successful. `optim`['par'] is a dictionary of point estimates,
            indexed by the parameter name. `optim`['value'] stores the value
            of the log-posterior (up to an additive constant, the ``lp__`` in
            Stan) corresponding to the point identified by `optim`['par'].

        Other parameters
        ----------------
        iter : int, optional
        epsilon : float, optional
        save_warmup : bool, optional
        refresh : int, optional

        Examples
        --------
        >>> from pystan import StanModel
        >>> m = StanModel(model_code='parameters {real y;} model {y ~ normal(0,1);}')
        >>> f = m.optimizing()

        """
        methods = ("BFGS", "Nesterov", "Newton")
        if method not in methods:
            raise ValueError("Method must be one of {}".format(methods))
        if data is None:
            data = {}

        data_r, data_i = pystan.misc._split_data(data)
        fit = self.fit_class(data_r, data_i)
        # store a copy of the data passed to fit in the class. The reason for
        # assigning data here rather than in StanFit4model's __cinit__
        # is that when __cinit__ is called there is no guarantee that the 
        # instance is ready for use as a normal Python instance. See the Cython
        # documentation on __cinit__.
        fit.data = {}
        fit.data.update(data_i)
        fit.data.update(data_r)

        m_pars = fit._get_param_names()
        p_dims = fit._get_param_dims()

        idx_of_lp = m_pars.index('lp__')
        del m_pars[idx_of_lp]
        del p_dims[idx_of_lp]

        if isinstance(init, Number):
            init = str(init)
        elif isinstance(init, Callable):
            init = init()
        elif not isinstance(init, Iterable) and \
                not isinstance(init, string_types):
            raise ValueError("Wrong specification of initial values.")

        if seed is None:
            seed = random.randint(0, MAX_UINT)
        seed = int(seed)

        stan_args = {'init': init, 'seed': seed}
        # methods: 1: newton; 2: nesterov; 3: bfgs
        stan_args['point_estimate'] = methods.index(method) + 1
        # set test gradient flag to false explicitly
        stan_args['test_grad'] = False
        stan_args.update(kwargs)

        stan_args = pystan.misc._get_valid_stan_args(stan_args)
        ret, sample = fit._call_sampler(stan_args)
        pars = pystan.misc._par_vector2dict(sample['par'], m_pars, p_dims)
        return OrderedDict([('par', pars),
                            ('value', sample['value'])])

    def sampling(self, data=None, pars=None, chains=4, iter=2000,
                 warmup=None, thin=1, seed=None,
                 init='random', sample_file=None, diagnostic_file=None,
                 verbose=False, **kwargs):
        """Draw samples from the model.

        Parameters
        ----------
        data : dict
            A Python dictionary providing the data for the model. Variables
            for Stan are stored in the dictionary as expected. Variable
            names are the keys and the values are their associated values.
            Stan only accepts certain kinds of values; see Notes.

        pars : list of string, optional
            A list of strings indicating parameters of interest. By default
            all parameters specified in the model will be stored.

        chains : int, optional
            Positive integer specifying number of chains. 4 by default.

        iter : int, 2000 by default
            Positive integer specifying how many iterations for each chain
            including warmup.

        warmup : int, iter//2 by default
            Positive integer specifying number of warmup (aka burin) iterations.
            As `warmup` also specifies the number of iterations used for step-size
            adaption, warmup samples should not be used for inference.

        thin : int, 1 by default
            Positive integer specifying the period for saving samples.

        seed : int, optional
            The seed, a positive integer for random number generation. Only
            one seed is needed when multiple chains are used, as the other
            chain's seeds are generated from the first chain's to prevent
            dependency among random number streams. By default, seed is
            ``random.randint(0, MAX_UINT)``.

        init : {0, '0', 'random', function returning dict, list of dict}, optional
            Specifies how initial parameter values are chosen: 0 or '0'
            initializes all to be zero on the unconstrained support; 'random'
            generates random initial values; list of size equal to the number
            of chains (`chains`), where the list contains a dict with initial
            parameter values; function returning a dict with initial parameter
            values. The function may take an optional argument `chain_id`.

        sample_file : string, optional
            File name specifying where samples for *all* parameters and other
            saved quantities will be written. If not provided, no samples
            will be written. If the folder given is not writable, a temporary
            directory will be used. When there are multiple chains, an underscore
            and chain number are appended to the file name. By default do not
            write samples to file.

        diagnostic_file : str, optional
            File name indicating where diagonstic data for all parameters
            should be written. If not writable, a temporary directory is used.

        verbose : boolean, False by default
            Indicates whether intermediate output should be piped to the
            console. This output may be useful for debugging.

        Returns
        -------
        fit : StanFit4<model_name>
            Instance containing the fitted results.

        Other parameters
        ----------------
        chain_id : int, optional
            Iterable of unique ints naming chains or int with which to start.
        leapfrog_steps : int, optional
        epsilon : float, optional
        gamma : float, optional
        delta : float, optional
        equal_step_sizes : bool, optional
        max_treedepth : int, optional
        nondiag_mass : bool, optional
        test_grad : bool
            If True, Stan will not perform any sampling. Instead the gradient
            calculation is tested and printed out and the fitted stanfit4model
            object will be in test gradient mode. False is the default.
        refresh : int, optional
            Controls how to indicate progress during sampling. By default,
            `refresh` = max(iter//10, 1).

        Notes
        -----

        More details can be found in Stan's manual. The default sampler is
        NUTS2, where `leapfrog_steps` is ``-1`` and `equal_step_sizes` is
        False. To use NUTS with full mass matrix, set `nondiag_mass` to True.

        Examples
        --------
        >>> from pystan import StanModel
        >>> m = StanModel(model_code='parameters {real y;} model {y ~ normal(0,1);}')
        >>> m.sampling(iter=100)

        """
        # NOTE: in this function, iter masks iter() the python function.
        # If this ever turns out to be a problem just add:
        # iter_ = iter
        # del iter  # now builtins.iter is available
        if diagnostic_file is not None:
            raise NotImplementedError("diagnostic_file not supported yet")
        if data is None:
            data = {}
        if warmup is None:
            warmup = int(iter // 2)

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

        if seed is None:
            seed = random.randint(0, MAX_UINT)
        seed = int(seed)

        args_list = pystan.misc._config_argss(chains=chains, iter=iter,
                                              warmup=warmup, thin=thin,
                                              init=init, seed=seed,
                                              sample_file=sample_file,
                                              diagnostic_file=diagnostic_file,
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
                logger.info(msg.format(mode, self.model_name, chain_num))
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
                    # rstan has this; name clashes with 'chains' in samples[0]['chains']
                   'chains': len(samples),
                   'iter': iter,
                   'warmup': warmup,
                   'thin': thin,
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
        fit.date = datetime.datetime.now()
        return fit
