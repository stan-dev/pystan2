#-----------------------------------------------------------------------------
# Copyright (c) 2013-2015, PyStan developers
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

from pystan._compat import PY2, string_types, implements_to_string, izip
from collections import OrderedDict
if PY2:
    from collections import Callable, Iterable
else:
    from collections.abc import Callable, Iterable
import datetime
import hashlib
import importlib
import imp
import io
import itertools
import logging
from numbers import Number
import os
import tempfile
import shutil
import string
import sys
import warnings

import distutils
from distutils.core import Extension

import Cython
from Cython.Build.Inline import _get_build_extension
from Cython.Build.Dependencies import cythonize

import numpy as np

import pystan.api
import pystan.misc

logger = logging.getLogger('pystan')


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


def _map_parallel(function, args, n_jobs):
    """multiprocessing.Pool(processors=n_jobs).map with some error checking"""
    # Following the error checking found in joblib
    multiprocessing = int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)) or None
    if multiprocessing:
        try:
            import multiprocessing
            import multiprocessing.pool
        except ImportError:
            multiprocessing = None
    # 2nd stage: validate that locking is available on the system and
    #            issue a warning if not
    if multiprocessing:
        try:
            _sem = multiprocessing.Semaphore()
            del _sem  # cleanup
        except (ImportError, OSError) as e:
            multiprocessing = None
            warnings.warn('%s. _map_parallel will operate in serial mode' % (e,))
    if multiprocessing and int(n_jobs) not in (0, 1):
        if n_jobs == -1:
            n_jobs = None
        pool = multiprocessing.Pool(processes=n_jobs)
        map_result = pool.map(function, args)
        pool.close()
        pool.join()
    else:
        map_result = list(map(function, args))
    return map_result


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
    module : builtins.module
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

    The Stan Development Team (2013) *Stan Modeling Language User's
    Guide and Reference Manual*.  <URL: http://mc-stan.org/>.

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
                 eigen_lib=None, verbose=False, obfuscate_model_name=True):

        if stanc_ret is None:
            stanc_ret = pystan.api.stanc(file=file,
                                         charset=charset,
                                         model_code=model_code,
                                         model_name=model_name,
                                         verbose=verbose,
                                         obfuscate_model_name=obfuscate_model_name)

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
        self.model_code = stanc_ret['model_code']
        self.model_cppcode = stanc_ret['cppcode']

        msg = "COMPILING THE C++ CODE FOR MODEL {} NOW."
        logger.info(msg.format(self.model_name))
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

        self._temp_dir = temp_dir = tempfile.mkdtemp()
        lib_dir = os.path.join(temp_dir, 'pystan')
        pystan_dir = os.path.dirname(__file__)
        include_dirs = [
            lib_dir,
            pystan_dir,
            os.path.join(pystan_dir, "stan", "src"),
            os.path.join(pystan_dir, "stan", "lib", "stan_math_2.8.0"),
            os.path.join(pystan_dir, "stan", "lib", "stan_math_2.8.0", "lib", "eigen_3.2.4"),
            os.path.join(pystan_dir, "stan", "lib", "stan_math_2.8.0", "lib", "boost_1.58.0"),
            np.get_include(),
        ]

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        model_cpp_file = os.path.join(lib_dir, self.model_cppname + '.hpp')
        with io.open(model_cpp_file, 'w', encoding='utf-8') as outfile:
            outfile.write(self.model_cppcode)

        pyx_file = os.path.join(lib_dir, module_name + '.pyx')
        pyx_template_file = os.path.join(pystan_dir, 'stanfit4model.pyx')
        with io.open(pyx_template_file, 'r', encoding='utf-8') as infile:
            s = infile.read()
            template = string.Template(s)
        with io.open(pyx_file, 'w', encoding='utf-8') as outfile:
            s = template.safe_substitute(model_cppname=self.model_cppname)
            outfile.write(s)

        stan_macros = [
            ('BOOST_RESULT_OF_USE_TR1', None),
            ('BOOST_NO_DECLTYPE', None),
            ('BOOST_DISABLE_ASSERTS', None),
            ('EIGEN_NO_DEBUG', None),
        ]
        extra_compile_args = [
            '-O0',
            '-ftemplate-depth-256',
            '-Wno-unused-function',
            '-Wno-uninitialized',
        ]

        distutils.log.set_verbosity(verbose)
        extension = Extension(name=module_name,
                              language="c++",
                              sources=[pyx_file],
                              define_macros=stan_macros,
                              include_dirs=include_dirs,
                              extra_compile_args=extra_compile_args)

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
        self.fit_class = getattr(self.module, "StanFit4Model")

    def __str__(self):
        # NOTE: returns unicode even for Python 2.7, implements_to_string
        # decorator creates __unicode__ and __str__
        s = u"StanModel object '{}' coded as follows:\n{}"
        return s.format(self.model_name, self.model_code)

    def __del__(self):
        _temp_dir = getattr(self, '_temp_dir', None)
        if _temp_dir:
            shutil.rmtree(_temp_dir, ignore_errors=True)

    def show(self):
        print(self)

    @property
    def dso(self):
        # warning added in PyStan 2.8.0
        warnings.warn('Accessing the module with `dso` is deprecated and will be removed in a future version. '\
                      'Use `module` instead.', DeprecationWarning)
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
        state['module_filename'] = state['module'].__file__
        state['module_name'] = state['module'].__name__
        if not os.path.exists(state['module_filename']):
            msg = 'Compiled module associated with Stan model not found at {}'.format(state['module_filename'])
            raise RuntimeError(msg)
        with io.open(state['module_filename'], 'rb') as f:
            state['module_bytes'] = f.read()
        del state['module']
        del state['fit_class']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # the following attributes are temporary and exist only in the
        # pickled object to facilitate reloading the module
        module_filename = self.module_filename
        module_bytes = self.module_bytes
        module_name = self.module_name
        del self.module_filename
        del self.module_bytes
        del self.module_name
        self._temp_dir = temp_dir = tempfile.mkdtemp()
        lib_dir = os.path.join(temp_dir, 'pystan')
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        module_basename = os.path.basename(module_filename)
        with io.open(os.path.join(lib_dir, module_basename), 'wb') as f:
            f.write(module_bytes)
        try:
            self.module = load_module(module_name, lib_dir)
            self.fit_class = getattr(self.module, "StanFit4Model")
        except Exception as e:
            logger.warning(e)
            logger.warning("Something went wrong while unpickling "
                            "the StanModel. Consider recompiling.")

    def optimizing(self, data=None, seed=None,
                   init='random', sample_file=None, algorithm=None,
                   verbose=False, as_vector=True, **kwargs):
        """Obtain a point estimate by maximizing the joint posterior.

        Parameters
        ----------
        data : dict
            A Python dictionary providing the data for the model. Variables
            for Stan are stored in the dictionary as expected. Variable
            names are the keys and the values are their associated values.
            Stan only accepts certain kinds of values; see Notes.

        seed : int or np.random.RandomState, optional
            The seed, a positive integer for random number generation. Only
            one seed is needed when multiple chains are used, as the other
            chain's seeds are generated from the first chain's to prevent
            dependency among random number streams. By default, seed is
            ``random.randint(0, MAX_UINT)``.

        init : {0, '0', 'random', function returning dict, list of dict}, optional
            Specifies how initial parameter values are chosen:
            - 0 or '0' initializes all to be zero on the unconstrained support.
            - 'random' generates random initial values. An optional parameter
              `init_r` controls the range of randomly generated initial values
              for parameters in terms of their unconstrained support;
            - list of size equal to the number of chains (`chains`), where the
              list contains a dict with initial parameter values;
            - function returning a dict with initial parameter values. The
              function may take an optional argument `chain_id`.

        sample_file : string, optional
            File name specifying where samples for *all* parameters and other
            saved quantities will be written. If not provided, no samples
            will be written. If the folder given is not writable, a temporary
            directory will be used. When there are multiple chains, an
            underscore and chain number are appended to the file name.
            By default do not write samples to file.

        algorithm : {"LBFGS", "BFGS", "Newton"}, optional
            Name of optimization algorithm to be used. Default is LBFGS.

        verbose : boolean, optional
            Indicates whether intermediate output should be piped to the console.
            This output may be useful for debugging. False by default.

        as_vector : boolean, optional
            Indicates an OrderedDict will be returned rather than a nested
            dictionary with keys 'par' and 'value'.

        Returns
        -------
        optim : OrderedDict
            Depending on `as_vector`, returns either an OrderedDict having
            parameters as keys and point estimates as values or an OrderedDict
            with components 'par' and 'value'.  ``optim['par']`` is a dictionary
            of point estimates, indexed by the parameter name.
            ``optim['value']`` stores the value of the log-posterior (up to an
            additive constant, the ``lp__`` in Stan) corresponding to the point
            identified by `optim`['par'].

        Other parameters
        ----------------
        iter : int, optional
            The maximum number of iterations.
        save_iterations : bool, optional
        refresh : int, optional
        init_alpha : float, optional
            For BFGS and LBFGS, see (Cmd)Stan manual. Default is 0.001
        tol_obj : float, optional
            For BFGS and LBFGS, see (Cmd)Stan manual. Default is 1e-12.
        tol_grad : float, optional
            For BFGS and LBFGS, see (Cmd)Stan manual. Default is 1e-8.
        tol_param : float, optional
            For BFGS and LBFGS, see (Cmd)Stan manual. Default is 1e-8.
        tol_rel_grad : float, optional
            For BFGS and LBFGS, see (Cmd)Stan manual. Default is 1e7.
        history_size : int, optional
            For LBFGS, see (Cmd)Stan manual. Default is 5.

        Examples
        --------
        >>> from pystan import StanModel
        >>> m = StanModel(model_code='parameters {real y;} model {y ~ normal(0,1);}')
        >>> f = m.optimizing()

        """
        algorithms = {"BFGS", "LBFGS", "Newton"}
        if algorithm is None:
            algorithm = "LBFGS"
        if algorithm not in algorithms:
            raise ValueError("Algorithm must be one of {}".format(algorithms))
        if data is None:
            data = {}

        fit = self.fit_class(data)

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

        seed = pystan.misc._check_seed(seed)

        stan_args = dict(init=init,
                         seed=seed,
                         method="optim",
                         algorithm=algorithm)
        if sample_file is not None:
            stan_args['sample_file'] = pystan.misc._writable_sample_file(sample_file)

        # check that arguments in kwargs are valid
        valid_args = {"iter", "save_iterations", "save_iterations", "refresh",
                      "init_alpha", "tol_obj", "tol_grad", "tol_param",
                      "tol_rel_obj", "tol_rel_grad", "history_size"}
        for arg in kwargs:
            if arg not in valid_args:
                raise ValueError("Parameter `{}` is not recognized.".format(arg))

        # This check is is to warn users of older versions of PyStan
        if kwargs.get('method'):
            raise ValueError('`method` is no longer used. Specify `algorithm` instead.')
        stan_args.update(kwargs)
        stan_args = pystan.misc._get_valid_stan_args(stan_args)

        ret, sample = fit._call_sampler(stan_args)
        pars = pystan.misc._par_vector2dict(sample['par'], m_pars, p_dims)
        if not as_vector:
            return OrderedDict([('par', pars), ('value', sample['value'])])
        else:
            return pars

    def sampling(self, data=None, pars=None, chains=4, iter=2000,
                 warmup=None, thin=1, seed=None, init='random',
                 sample_file=None, diagnostic_file=None, verbose=False,
                 algorithm=None, control=None, n_jobs=-1, **kwargs):
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

        seed : int or np.random.RandomState, optional
            The seed, a positive integer for random number generation. Only
            one seed is needed when multiple chains are used, as the other
            chain's seeds are generated from the first chain's to prevent
            dependency among random number streams. By default, seed is
            ``random.randint(0, MAX_UINT)``.

        algorithm : {"NUTS", "HMC", "Fixed_param"}, optional
            One of algorithms that are implemented in Stan such as the No-U-Turn
            sampler (NUTS, Hoffman and Gelman 2011), static HMC, or ``Fixed_param``.

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

        control : dict, optional
            A dictionary of parameters to control the sampler's behavior. Default
            values are used if control is not specified.  The following are
            adaptation parameters for sampling algorithms.

            These are parameters used in Stan with similar names:

            - `adapt_engaged` : bool, default True
            - `adapt_gamma` : float, positive, default 0.05
            - `adapt_delta` : float, between 0 and 1, default 0.8
            - `adapt_kappa` : float, between default 0.75
            - `adapt_t0`    : float, positive, default 10

            In addition, the algorithm HMC (called 'static HMC' in Stan) and NUTS
            share the following parameters:

            - `stepsize`: float, positive
            - `stepsize_jitter`: float, between 0 and 1
            - `metric` : str, {"unit_e", "diag_e", "dense_e"}

            In addition, depending on which algorithm is used, different parameters
            can be set as in Stan for sampling. For the algorithm HMC we can set

            - `int_time`: float, positive

            For algorithm NUTS, we can set

            - `max_treedepth` : int, positive

        n_jobs : int, optional
            Sample in parallel. If -1 all CPUs are used. If 1, no parallel
            computing code is used at all, which is useful for debugging.

        Returns
        -------
        fit : StanFit4Model
            Instance containing the fitted results.

        Other parameters
        ----------------

        chain_id : int or iterable of int, optional
            `chain_id` can be a vector to specify the chain_id for all chains or
            an integer. For the former case, they should be unique. For the latter,
            the sequence of integers starting from the given `chain_id` are used
            for all chains.

        init_r : float, optional
            `init_r` is only valid if `init` == "random". In this case, the intial
            values are simulated from [-`init_r`, `init_r`] rather than using the
            default interval (see the manual of Stan).

        test_grad: bool, optional

        append_samples`: bool, optional

        refresh`: int, optional
            Argument `refresh` can be used to control how to indicate the progress
            during sampling (i.e. show the progress every \code{refresh} iterations).
            By default, `refresh` is `max(iter/10, 1)`.

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
        algorithms = ("NUTS", "HMC", "Fixed_param")  # , "Metropolis")
        algorithm = "NUTS" if algorithm is None else algorithm
        if algorithm not in algorithms:
            raise ValueError("Algorithm must be one of {}".format(algorithms))

        fit = self.fit_class(data)

        m_pars = fit._get_param_names()
        p_dims = fit._get_param_dims()

        if isinstance(pars, string_types):
            pars = [pars]
        if pars is not None and len(pars) > 0:
            fit._update_param_oi(pars)
            if not all(p in m_pars for p in pars):
                pars = np.asarray(pars)
                unmatched = pars[np.invert(np.in1d(pars, m_pars))]
                msg = "No parameter(s): {}; sampling not done."
                raise ValueError(msg.format(', '.join(pars[unmatched])))

        if chains < 1:
            raise ValueError("The number of chains is less than one; sampling"
                             "not done.")

        # check that arguments in kwargs are valid
        valid_args = {"chain_id", "init_r", "test_grad", "append_samples", "refresh", "control"}
        for arg in kwargs:
            if arg not in valid_args:
                raise ValueError("Parameter `{}` is not recognized.".format(arg))

        args_list = pystan.misc._config_argss(chains=chains, iter=iter,
                                              warmup=warmup, thin=thin,
                                              init=init, seed=seed, sample_file=sample_file,
                                              diagnostic_file=diagnostic_file,
                                              algorithm=algorithm,
                                              control=control, **kwargs)

        # number of samples saved after thinning
        warmup2 = 1 + (warmup - 1) // thin
        n_kept = 1 + (iter - warmup - 1) // thin
        n_save = n_kept + warmup2

        if n_jobs is None:
            n_jobs = -1

        # disable multiprocessing if we only have a single chain
        if chains == 1:
            n_jobs = 1

        assert len(args_list) == chains
        call_sampler_args = izip(itertools.repeat(data), args_list)
        call_sampler_star = self.module._call_sampler_star
        ret_and_samples = _map_parallel(call_sampler_star, call_sampler_args, n_jobs)
        samples = [smpl for _, smpl in ret_and_samples]

        # _organize_inits strips out lp__ (RStan does it in this method)
        inits_used = pystan.misc._organize_inits([s['inits'] for s in samples], m_pars, p_dims)

        random_state = np.random.RandomState(args_list[0]['seed'])
        perm_lst = [random_state.permutation(int(n_kept)) for _ in range(chains)]
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
                   'dims_oi': fit._get_param_dims_oi(),
                   'fnames_oi': fnames_oi,
                   'n_flatnames': n_flatnames}
        fit.model_name = self.model_name
        fit.model_pars = m_pars
        fit.par_dims = p_dims
        fit.mode = 0 if not kwargs.get('test_grad') else 1
        fit.inits = inits_used
        fit.stan_args = args_list
        fit.stanmodel = self
        fit.date = datetime.datetime.now()
        return fit
