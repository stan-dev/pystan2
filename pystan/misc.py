"""PyStan utility functions

These functions validate and organize data passed to and from the
classes and functions defined in the file `stan_fit.hpp` and wrapped
by the Cython file `stan_fit.pxd`.

"""
#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

# REF: rstan/rstan/R/misc.R

from pystan._compat import PY2, string_types
from collections import OrderedDict
if PY2:
    from collections import Callable, Sequence
else:
    from collections.abc import Callable, Sequence
import inspect
import io
import logging
from numbers import Number
import os
import random
import sys
import time

import numpy as np

from pystan.constants import MAX_UINT

def _split_data(data):
    data_r = {}
    data_i = {}
    # data_r and data_i are going to be converted into C++ objects of
    # type: map<string, pair<vector<double>, vector<size_t>>> and
    # map<string, pair<vector<int>, vector<size_t>>> so prepare
    # them accordingly.
    for k, v in data.items():
        if np.issubdtype(np.asarray(v).dtype, int):
            data_i.update({k.encode('utf-8'): np.asarray(v, dtype=int)})
        elif np.issubdtype(np.asarray(v).dtype, float):
            data_r.update({k.encode('utf-8'): np.asarray(v, dtype=float)})
        else:
            msg = "Variable {} neither int nor float (dtype: {})"
            raise ValueError(msg.format(k, v.dtype))
    return data_r, data_i


def _config_argss(chains, iter, warmup, thin, init, seed, sample_file,
                  diagnostic_file, **kwargs):
    iter = int(iter)
    if iter < 1:
        raise ValueError("`iter` should be a positive integer.")
    thin = int(thin)
    if thin < 1 or thin > iter:
        raise ValueError("`thin should be a positive integer "
                         "less than `iter`.")
    warmup = max(0, int(warmup))
    if warmup > iter:
        raise ValueError("`warmup` should be an integer less than `iter`.")
    chains = int(chains)
    if chains < 1:
        raise ValueError("`chains` should be a positive integer.")

    iters = [iter] * chains
    thins = [thin] * chains
    warmups = [warmup] * chains

    inits_specified = False
    # slight difference here from rstan; Python's lists are not typed.
    if isinstance(init, Number):
        init = str(init)
    if isinstance(init, string_types):
        if init in ['0', 'random']:
            inits = [init] * chains
        else:
            inits = ["random"] * chains
        inits_specified = True
    if not inits_specified and isinstance(init, Callable):
        ## test if function takes argument named "chain_id"
        if "chain_id" in inspect.getargspec(init).args:
            inits = [init(chain_id=id) for id in range(chains)]
        else:
            inits = [init()] * chains
        if not isinstance(inits[0], dict):
            raise ValueError("The function specifying initial values must "
                             "return a dictionary.")
        inits_specified = True
    if not inits_specified and isinstance(init, Sequence):
        if len(init) != chains:
            raise ValueError("Length of list of initial values does not "
                             "match number of chains.")
        if not all([isinstance(d, dict) for d in init]):
            raise ValueError("Initial value list is not a sequence of "
                             "dictionaries.")
        inits = init
        inits_specified = True
    if not inits_specified:
        raise ValueError("Invalid specification of initial values.")

    ## only one seed is needed by virtue of the RNG
    if seed is None:
        seed = random.randint(0, MAX_UINT)
    else:
        seed = _check_seed(seed)

    kwargs['point_estimate'] = -1

    # use chain_id argument if specified
    if kwargs.get('chain_id') is None:
        chain_ids = list(range(1, chains + 1))
    else:
        chain_id = [int(id) for id in kwargs['chain_id']]
        if len(set(chain_id)) != len(chain_id):
            raise ValueError("`chain_id` has duplicated elements.")
        chain_id_len = len(chain_id)
        if chain_id_len >= chains:
            chain_ids = chain_id
        else:
            chain_ids = chain_id + [max(chain_id) + 1 + i
                                    for i in range(chains - chain_id_len)]
        kwargs['chain_id'] = None  # NOTE: in rstan, not sure why

    argss = [dict() for _ in range(chains)]
    for i in range(chains):
        argss[i] = {'chain_id': chain_ids[i], 'iter': iters[i],
                    'thin': thins[i], 'seed': seed,
                    'warmup': warmups[i], 'init': inits[i]}

    if sample_file is not None:
        # FIXME: to implement
        raise NotImplementedError
    
    if diagnostic_file is not None:
        raise NotImplementedError("diagnostic_file not implemented yet.")

    for i in range(chains):
        argss[i].update(kwargs)
        argss[i] = _get_valid_stan_args(argss[i])

    return argss


def _get_valid_stan_args(base_args=None):
    """Fill in default values for arguments not provided in `base_args`.

    RStan does this in cpp in stan_args.hpp. It seems easier to deal
    with here in Python.

    """
    args = base_args.copy() if base_args is not None else {}
    # Default arguments, c.f. rstan/rstan/inst/include/rstan/stan_args.hpp
    # values in args are going to be converted into C++ objects so
    # prepare them accordingly---e.g., bytes -> std::string
    args['save_warmup'] = args.get('save_warmup', True)
    args['sample_file_flag'] = args.get('sample_file_flag', False)
    args['diagnostic_file'] = args.get('diagnostic_file', '')
    if args['diagnostic_file']:
        args['diagnostic_file_flag'] = True
    else:
        args['diagnostic_file_flag'] = False
    args['iter'] = args.get('iter', 2000)
    args['warmup'] = args.get('warmup', args['iter'] // 2)
    args['thin'] = args.get('thin', (args['iter'] - args['warmup']) // 2)
    args['iter_save_wo_warmup'] = args.get('iter_save_wo_warmup',
                        1 + (args['iter']-args['warmup']-1) // args['thin'])
    args['iter_save'] = args.get('iter_save',
                                 args['iter_save_wo_warmup'] + 1 +
                                 (args['warmup'] - 1) // args['thin'])
    args['leapfrog_steps'] = args.get('leapfrog_steps', -1)
    args['epsilon'] = args.get('epsilon', -1)
    args['epsilon_pm'] = args.get('epsilon_pm', 0.0)
    args['max_treedepth'] = args.get('max_treedepth', 10)
    args['equal_step_sizes'] = args.get('equal_step_sizes', False)
    args['delta'] = args.get('delta', 0.5)
    args['gamma'] = args.get('gamma', 0.05)
    args['refresh'] = args.get('refresh',
                               args['iter'] // 10 if args['iter'] >= 20 else 1)
    # NB: argument named "seed" not "random_seed"
    args['random_seed'] = args.get('seed', int(time.time()))
    args['random_seed_src'] = args.get('random_seed_src',
                                       "random").encode('utf-8')
    args['chain_id'] = args.get('chain_id', 1)
    args['chain_id_src'] = args.get('chain_id_src',
                                    "default").encode('utf-8')
    args['init'] = args.get('init', "random").encode('utf-8')
    args['append_samples'] = args.get('append_samples', False)
    args['test_grad'] = args.get('test_grad', False)
    args['nondiag_mass'] = args.get('nondiag_mass', False)
    args['point_estimate'] = args.get('point_estimate', -1)
    return args


def _check_seed(seed, raise_exception=True):
    if isinstance(seed, string_types):
        try:
            seed = int(seed)
        except ValueError:
            if raise_exception is True:
                raise ValueError("`seed` must be a string of digits.")
            else:
                logging.warn("`seed` needs to be a string of digits.")
                return None
    elif isinstance(seed, Number):
        seed = int(seed)
    elif seed is None:
        seed = random.randint(0, MAX_UINT)
    else:
        logging.warn('`seed` did not have expected characteristics.')

    return seed


def _organize_inits(inits, pars, dims):
    """Obtain a list of initial values for each chain.

    The parameter 'lp__' will be removed from the chains.

    Parameters
    ----------
    inits : list
        list of initial values for each chain.
    pars : list of str
    dims : list of list of int
        from (via cython conversion) vector[vector[uint]] dims

    Returns
    -------
    inits : list of dict

    """
    try:
        idx_of_lp = pars.index('lp__')
        del pars[idx_of_lp]
        del dims[idx_of_lp]
    except ValueError:
        pass
    starts = _calc_starts(dims)
    return [_par_vector2dict(init, pars, dims, starts) for init in inits]


def _calc_starts(dims):
    """Calculate starting indexes

    Parameters
    ----------
    dims : list of list of int
        from (via cython conversion) vector[vector[uint]] dims

    Examples
    --------
    >>> _calc_starts([[8, 2], [5], [6, 2]])
    [0, 16, 21]

    """
    # NB: Python uses 0-indexing; R uses 1-indexing.
    l = len(dims)
    s = [np.prod(d) for d in dims]
    starts = np.cumsum([0] + s)[0:l].tolist()
    # coerce things into ints before returning
    return [int(s) for s in starts]


def _par_vector2dict(v, pars, dims, starts=None):
    """Turn a vector of samples into an OrderedDict according to param dims.

    Parameters
    ----------
    y : list of int or float
    pars : list of str
        parameter names
    dims : list of list of int
        list of dimensions of parameters

    Returns
    -------
    d : dict

    Examples
    --------
    >>> v = list(range(31))
    >>> dims = [[5], [5, 5], []]
    >>> pars = ['mu', 'Phi', 'eta']
    >>> _par_vector2dict(v, pars, dims)  # doctest: +ELLIPSIS
    OrderedDict([('mu', array([0, 1, 2, 3, 4])), ('Phi', array([[ 5, ...

    """
    if starts is None:
        starts = _calc_starts(dims)
    d = OrderedDict()
    for i in range(len(pars)):
        l = int(np.prod(dims[i]))
        start = starts[i]
        end = start + l
        y = np.asarray(v[start:end])
        if len(dims[i]) > 1:
            y = y.reshape(dims[i], order='F')  # 'F' = Fortran, column-major
        # squeeze deals with scalars (1,)
        d[pars[i]] = y.squeeze() if len(dims[i]) == 0 else y
    return d


def _check_pars(allpars, pars):
    if len(pars) == 0:
        raise ValueError("No parameter specified (`pars` is empty).")
    for par in pars:
        if par not in allpars:
            raise ValueError("No parameter {}".format(par))


def _pars_total_indexes(names, dims, fnames, pars):
    """Obtain all the indexes for parameters `pars` in the sequence of names.

    `names` references variables that are in column-major order

    Parameters
    ----------
    names : sequence of str
        All the parameter names.
    dim : sequence of list of int
        Dimensions, in same order as `names`.
    fnames : sequence of str
        All the scalar parameter names
    pars : sequence of str
        The parameters of interest. It is assumed all elements in `pars` are in
        `names`.

    Returns
    -------
    indexes : OrderedDict of list of int
        Dictionary uses parameter names as keys.

    Note
    ----
    Inside each parameter (vector or array), the sequence uses column-major
    ordering. For example, if we have parameters alpha and beta, having
    dimensions [2, 2] and [2, 3] respectively, the whole parameter sequence
    is alpha[0,0], alpha[1,0], alpha[0, 1], alpha[1, 1], beta[0, 0],
    beta[1, 0], beta[0, 1], beta[1, 1], beta[0, 2], beta[1, 2]. In short,
    like R matrix(..., bycol=TRUE).

    Example
    -------
    >>> pars_oi = ['mu', 'tau', 'eta', 'theta', 'lp__']
    >>> dims_oi = [[], [], [8], [8], []]
    >>> fnames_oi = ['mu', 'tau', 'eta[1]', 'eta[2]', 'eta[3]', 'eta[4]',
    ... 'eta[5]', 'eta[6]', 'eta[7]', 'eta[8]', 'theta[1]', 'theta[2]',
    ... 'theta[3]', 'theta[4]', 'theta[5]', 'theta[6]', 'theta[7]',
    ... 'theta[8]', 'lp__']
    >>> pars = ['mu', 'tau', 'eta', 'theta', 'lp__']
    >>> _pars_total_indexes(pars_oi, dims_oi, fnames_oi, pars)
    ... # doctest: +ELLIPSIS
    OrderedDict([('mu', (0,)), ('tau', (1,)), ('eta', (2, 3, ...

    """
    starts = _calc_starts(dims)

    def par_total_indexes(par):
        # if `par` is a scalar, it will match one of `fnames`
        if par in fnames:
            p = fnames.index(par)
            return {par: tuple([p])}
        else:
            p = names.index(par)
            indexes = starts[p] + np.arange(np.prod(dims[p]))
            return {par: tuple(indexes)}

    indexes = OrderedDict()
    for par in pars:
        indexes.update(par_total_indexes(par))
    return indexes


def _get_kept_samples(n, sim):
    # NOTE: this is in stanfit-class.R in RStan (rather than misc.R)
    """Get samples to be kept from the chain(s) for `n`th parameter.

    Samples from different chains are merged.

    Parameters
    ----------
    n : int
    sim : dict
        A dictionary tied to a StanFit4<model> instance.

    Returns
    -------
    samples : array
        Samples being kept, permuted and in column-major order.

    """
    ss = []
    for s, nw, perm in zip(sim['samples'], sim['warmup2'], sim['permutation']):
        nth_key = list(s['chains'].keys())[n]
        r = s['chains'][nth_key][nw:]
        ss.extend(r[perm])
    return np.asarray(ss)


def _get_samples(n, sim, inc_warmup=True):
    # NOTE: this is in stanfit-class.R in RStan (rather than misc.R)
    """Get chain for `n`th parameter.

    Parameters
    ----------
    n : int
    sim : dict
        A dictionary tied to a StanFit4<model> instance.

    Returns
    -------
    chains : list of array
        Each chain is an element in the list.

    """
    if all(w2 == 0 for w2 in sim['warmup2']):
        # from RStan
        inc_warmup = True
    ss = []
    for s, nw in zip(sim['samples'], sim['warmup2']):
        nth_key = list(s['chains'].keys())[n]
        r = s['chains'][nth_key] if inc_warmup else s['chains'][nth_key][nw:]
        ss.append(np.asarray(r))
    return ss

def _redirect_stderr():
    """Redirect stderr for subprocesses to /dev/null

    Silences copious compilation messages.

    Returns
    -------
    orig_stderr : file descriptor
        Copy of original stderr file descriptor
    """
    sys.stderr.flush()
    stderr_fileno = sys.stderr.fileno()
    orig_stderr = os.dup(stderr_fileno)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fileno)
    os.close(devnull)
    return orig_stderr

def _has_fileno(stream):
    """Returns whether the stream object seems to have a working fileno()

    Tells whether _redirect_stderr is likely to work.

    Parameters
    ----------
    stream : IO stream object

    Returns
    -------
    has_fileno : bool
        True if stream.fileno() exists and doesn't raise OSError or
        UnsupportedOperation
    """
    try:
        stream.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation):
        return False
    return True
