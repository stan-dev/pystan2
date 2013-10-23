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

from __future__ import unicode_literals, division
from pystan._compat import PY2, string_types

from collections import OrderedDict
if PY2:
    from collections import Callable, Sequence
else:
    from collections.abc import Callable, Sequence
import inspect
import io
import itertools
import logging
from numbers import Number
import os
import random
import re
import sys
import tempfile
import time

import numpy as np
try:
    from scipy.stats.mstats import mquantiles
except ImportError:
    from pystan.external.mstats import mquantiles

import pystan.chains as chains
from pystan.constants import (MAX_UINT, sampling_algo_t, optim_algo_t,
                              sampling_metric_t, stan_args_method_t)


logger = logging.getLogger('pystan')
logger.setLevel(logging.INFO)


def _print_stanfit(fit, pars=None, probs=(0.025, 0.25, 0.5, 0.75, 0.975),
                   digits_summary=1):
        if fit.mode == 1:
            return "Stan model '{}' is of mode 'test_grad';\n"\
                   "sampling is not conducted.".format(fit.model_name)
        elif fit.mode == 2:
            return "Stan model '{}' does not contain samples.".format(fit.model_name)
        if pars is None:
            pars = fit.sim['pars_oi']
            fnames = fit.sim['fnames_oi']
        else:
            # FIXME: does this case ever occur?
            # need a way of getting fnames matching specified pars
            raise NotImplementedError

        n_kept = [s - w for s, w in zip(fit.sim['n_save'], fit.sim['warmup2'])]
        header = "Inference for Stan model: {}.\n".format(fit.model_name)
        header += "{} chains, each with iter={}; warmup={}; thin={}; \n"
        header = header.format(fit.sim['chains'], fit.sim['iter'], fit.sim['warmup'],
                               fit.sim['thin'], sum(n_kept))
        header += "post-warmup draws per chain={}, total post-warmup draws={}.\n\n"
        header = header.format(n_kept[0], sum(n_kept))
        footer = "\n\nSamples were drawn using {} at {}.\n"\
            "For each parameter, n_eff is a crude measure of effective sample size,\n"\
            "and Rhat is the potential scale reduction factor on split chains (at \n"\
            "convergence, Rhat=1)."
        sampler = fit.sim['samples'][0]['args']['sampler_t']
        date = fit.date.strftime('%c')  # %c is locale's representation
        footer = footer.format(sampler, date)
        s = _summary(fit, pars, probs)
        body = _array_to_table(s['summary'], s['summary_rownames'],
                               s['summary_colnames'], digits_summary)
        return header + body + footer


def _array_to_table(arr, rownames, colnames, n_digits):
    """Print an array with row and column names

    Example:
                  mean se_mean  sd 2.5%  25%  50%  75% 97.5% n_eff Rhat
        beta[1,1]  0.0     0.0 1.0 -2.0 -0.7  0.0  0.7   2.0  4000    1
        beta[1,2]  0.0     0.0 1.0 -2.1 -0.7  0.0  0.7   2.0  4000    1
        beta[2,1]  0.0     0.0 1.0 -2.0 -0.7  0.0  0.7   2.0  4000    1
        beta[2,2]  0.0     0.0 1.0 -1.9 -0.6  0.0  0.7   2.0  4000    1
        lp__      -4.2     0.1 2.1 -9.4 -5.4 -3.8 -2.7  -1.2   317    1
    """
    assert arr.shape == (len(rownames), len(colnames))
    rownames_maxwidth = max(len(n) for n in rownames)
    widths = [rownames_maxwidth] + [max(5, len(n) + 1) for n in colnames]
    header = '{:>{width}}'.format('', width=widths[0])
    for name, width in zip(colnames, widths[1:]):
        header += '{name:>{width}}'.format(name=name, width=width)
    lines = [header]
    for rowname, row in zip(rownames, arr):
        line = '{name:{width}}'.format(name=rowname, width=widths[0])
        for j, (num, width) in enumerate(zip(row, widths[1:])):
            if colnames[j] == 'n_eff':
                num = int(round(num, 0))
            line += '{num:{width}}'.format(num=round(num, n_digits), width=width)
        lines.append(line)
    return '\n'.join(lines)


def _summary(fit, pars=None, probs=None, **kwargs):
    """Summarize samples (compute mean, SD, quantiles) in all chains.

    REF: stanfit-class.R summary method

    Parameters
    ----------
    fit : stanfit4model object
    pars : str or sequence of str, optional
        Parameter names. By default use all parameters
    probs : sequence of float, optional
        Quantiles. By default, (0.025, 0.25, 0.5, 0.75, 0.975)

    Returns
    -------
    summaries : OrderedDict of array
        Array indexed by 'summary' has dimensions (num_params, num_statistics).
        Parameters are unraveled in *row-major order*. Statistics include: mean,
        se_mean, sd, probs_0, ..., probs_n, n_eff, and Rhat. Array indexed by
        'c_summary' breaks down the statistics by chain and has dimensions
        (num_params, num_statistics_c_summary, num_chains). Statistics for
        `c_summary` are the same as for `summary` with the exception that
        se_mean, n_eff, and Rhat are absent. Row names and column names are
        also included in the OrderedDict.
    """
    if fit.mode == 1:
        msg = "Stan model {} is of mode 'test_grad'; sampling is not conducted."
        msg = msg.format(fit.model_name)
        raise ValueError(msg)
    elif fit.mode == 2:
        msg = "Stan model {} contains no samples.".format(fit.model_name)
        raise ValueError(msg)

    if fit.sim['n_save'] == fit.sim['warmup2']:
        msg = "Stan model {} contains no samples.".format(fit.model_name)
        raise ValueError(msg)

    # rstan checks for cached summaries here

    if pars is None:
        pars = fit.sim['pars_oi']
    elif isinstance(pars, string_types):
        pars = [pars]
    if probs is None:
        probs = (0.025, 0.25, 0.5, 0.75, 0.975)
    ss = _summary_sim(fit.sim, pars, probs)
    # TODO: include sem, ess and rhat: ss['ess'], ss['rhat']
    s1 = np.column_stack([ss['msd'][:, 0], ss['sem'], ss['msd'][:, 1], ss['quan'], ss['ess'], ss['rhat']])
    s1_rownames = ss['c_msd_names']['parameters']
    s1_colnames = (ss['c_msd_names']['stats'][0],) + ('se_mean',) + \
            (ss['c_msd_names']['stats'][1],) + ss['c_quan_names']['stats'] + \
            ('n_eff', 'Rhat')
    s2 = _combine_msd_quan(ss['c_msd'], ss['c_quan'])
    s2_rownames = ss['c_msd_names']['parameters']
    s2_colnames = ss['c_msd_names']['stats'] + ss['c_quan_names']['stats']
    return OrderedDict(summary=s1, c_summary=s2,
                       summary_rownames=s1_rownames,
                       summary_colnames=s1_colnames,
                       c_summary_rownames=s2_rownames,
                       c_summary_colnames=s2_colnames)


def _combine_msd_quan(msd, quan):
    """Combine msd and quantiles in chain summary

    Parameters
    ----------
    msd : array of shape (num_params, 2, num_chains)
       mean and sd for chains
    cquan : array of shape (num_params, num_quan, num_chains)
        quantiles for chains

    Returns
    -------
    msdquan : array of shape (num_params, 2 + num_quan, num_chains)
    """
    dim1 = msd.shape
    dim2 = quan.shape
    n_par, _, n_chains = dim1
    ll = []
    for i in range(n_chains):
        a1 = msd[:, :, i]
        a2 = quan[:, :, i]
        ll.append(np.column_stack([a1, a2]))
    msdquan = np.dstack(ll)
    return msdquan


def _summary_sim(sim, pars, probs):
    """Summarize chains together and separately

    REF: rstan/rstan/R/misc.R

    Parameters are unraveled in *column-major order*.

    Parameters
    ----------
    sim : dict
        dict from from a stanfit fit object, i.e., fit['sim']
    pars : Iterable of str
        parameter names
    probs : Iterable of probs
        desired quantiles

    Returns
    -------
    summaries : OrderedDict of array
        This dictionary contains the following arrays indexed by the keys
        given below:
        - 'msd' : array of shape (num_params, 2) with mean and sd
        - 'sem' : array of length num_params with standard error for the mean
        - 'c_msd' : array of shape (num_params, 2, num_chains)
        - 'quan' : array of shape (num_params, num_quan)
        - 'c_quan' : array of shape (num_params, num_quan, num_chains)
        - 'ess' : array of shape (num_params, 1)
        - 'rhat' : array of shape (num_params, 1)

    Note
    ----
    `_summary_sim` has the parameters in *column-major* order whereas `_summary`
    gives them in *row-major* order. (This follows RStan.)
    """
    # NOTE: this follows RStan rather closely. Some of the calculations here
    probs_len = len(probs)
    n_chains = len(sim['samples'])
    # tidx is a dict with keys that are parameters and values that are their
    # indices using column-major ordering
    tidx = _pars_total_indexes(sim['pars_oi'], sim['dims_oi'], sim['fnames_oi'], pars)
    tidx_colm = [tidx[par] for par in pars]
    tidx_colm = list(itertools.chain(*tidx_colm))  # like R's unlist()
    tidx_rowm = [tidx[par+'_rowmajor'] for par in pars]
    tidx_rowm = list(itertools.chain(*tidx_rowm))
    tidx_len = len(tidx_colm)
    lmsdq = [_get_par_summary(sim, i, probs) for i in tidx_colm]
    msd = np.row_stack([x['msd'] for x in lmsdq])
    quan = np.row_stack([x['quan'] for x in lmsdq])
    probs_str = tuple(["{:g}%".format(100*p) for p in probs])
    msd.shape = (tidx_len, 2)
    quan.shape = (tidx_len, probs_len)

    c_msd = np.row_stack([x['c_msd'] for x in lmsdq])
    c_quan = np.row_stack([x['c_quan'] for x in lmsdq])
    c_msd.shape = (tidx_len, 2, n_chains)
    c_quan.shape = (tidx_len, probs_len, n_chains)
    sim_attr_args = sim.get('args', None)
    if sim_attr_args is None:
        cids = list(range(n_chains))
    else:
        cids = [x['chain_id'] for x in sim_attr_args]

    c_msd_names = dict(parameters=np.asarray(sim['fnames_oi'])[tidx_colm],
                       stats=("mean", "sd"),
                       chains=tuple("chain:{}".format(cid) for cid in cids))
    c_quan_names = dict(parameters=np.asarray(sim['fnames_oi'])[tidx_colm],
                        stats=probs_str,
                        chains=tuple("chain:{}".format(cid) for cid in cids))
    # TODO: include sem, ess and rhat, see rstan/rstan/src/chains.cpp
    ess = np.array([chains.ess(sim, n) for n in tidx_colm], dtype=int)
    rhat = np.array([chains.splitrhat(sim, n) for n in tidx_colm])
    return dict(msd=msd, c_msd=c_msd, c_msd_names=c_msd_names, quan=quan,
                c_quan=c_quan, c_quan_names=c_quan_names,
                sem=msd[:, 1] / np.sqrt(ess), ess=ess, rhat=rhat,
                row_major_idx=tidx_rowm, col_major_idx=tidx_colm)


def _get_par_summary(sim, n, probs):
    """Summarize chains merged and individually

    Parameters
    ----------
    sim : dict from stanfit object
    n : int
        parameter index
    probs : iterable of int
        quantiles

    Returns
    -------
    summary : dict
       Dictionary containing summaries
    """
    # _get_samples gets chains for nth parameter
    ss = _get_samples(n, sim, inc_warmup=False)
    msdfun = lambda chain: (np.mean(chain), np.std(chain, ddof=1))
    qfun = lambda chain: mquantiles(chain, probs)
    c_msd = np.array([msdfun(s) for s in ss]).flatten()
    c_quan = np.array([qfun(s) for s in ss]).flatten()
    ass = np.asarray(ss).flatten()
    msd = np.asarray(msdfun(ass))
    quan = qfun(np.asarray(ass))
    return dict(msd=msd, quan=quan, c_msd=c_msd, c_quan=c_quan)


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


def _config_argss(chains, iter, warmup, thin,
                  init, seed, sample_file, diagnostic_file, algorithm,
                  control, **kwargs):
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
        del kwargs['chain_id']

    kwargs['method'] = "test_grad" if kwargs.get('test_grad') else 'sampling'

    all_metrics = ("unit_e", "diag_e", "dense_e")
    if control is not None:
        if not isinstance(control, dict):
            raise ValueError("control must be a dictionary")
        metric = control.get('metric')
        if metric is not None:
            if metric not in all_metrics:
                raise ValueError("Metric must be one of {}".format(all_metrics))
            control['metric'] = metric
        kwargs['control'] = control

    argss = [dict() for _ in range(chains)]
    for i in range(chains):
        argss[i] = dict(chain_id=chain_ids[i],
                        iter=iters[i], thin=thins[i], seed=seed,
                        warmup=warmups[i], init=inits[i],
                        algorithm=algorithm)

    if sample_file is not None:
        sample_file = _writable_sample_file(sample_file)
        if chains == 1:
            argss[0]['sample_file'] = sample_file
        elif chains > 1:
            for i in range(chains):
                argss[i]['sample_file'] = _append_id(sample_file, i)

    if diagnostic_file is not None:
        raise NotImplementedError("diagnostic_file not implemented yet.")

    for i in range(chains):
        argss[i].update(kwargs)
        argss[i] = _get_valid_stan_args(argss[i])

    return argss


def _get_valid_stan_args(base_args=None):
    """Fill in default values for arguments not provided in `base_args`.

    RStan does this in C++ in stan_args.hpp in the stan_args constructor.
    It seems easier to deal with here in Python.

    """
    args = base_args.copy() if base_args is not None else {}
    # Default arguments, c.f. rstan/rstan/inst/include/rstan/stan_args.hpp
    # values in args are going to be converted into C++ objects so
    # prepare them accordingly---e.g., unicode -> bytes -> std::string
    args['chain_id'] = args.get('chain_id', 1)
    args['append_samples'] = args.get('append_samples', False)
    if args.get('method') is None or args['method'] == "sampling":
        args['method'] = stan_args_method_t.SAMPLING
    elif args['method'] == "optim":
        args['method'] = stan_args_method_t.OPTIM
    elif args['method'] == 'test_grad':
        args['method'] = stan_args_method_t.TEST_GRADIENT
    else:
        args['method'] = stan_args_method_t.SAMPLING
    args['sample_file_flag'] = True if args.get('sample_file') else False
    args['sample_file'] = args.get('sample_file', '').encode('ascii')
    args['diagnostic_file_flag'] = True if args.get('diagnostic_file') else False
    args['diagnostic_file'] = args.get('diagnostic_file', '').encode('ascii')

    if args['method'] == stan_args_method_t.SAMPLING:
        args['ctrl'] = args.get('ctrl', dict(sampling=dict()))
        args['ctrl']['sampling']['iter'] = iter = args.get('iter', 2000)
        args['ctrl']['sampling']['warmup'] = warmup = args.get('warmup', args['iter'] // 2)
        calculated_thin = iter - warmup // 1000
        if calculated_thin < 1:
            calculated_thin = 1
        args['ctrl']['sampling']['thin'] = thin = args.get('thin', calculated_thin)
        args['ctrl']['sampling']['save_warmup'] = True  # always True now
        args['ctrl']['sampling']['iter_save_wo_warmup'] = iter_save_wo_warmup = 1 + (iter - warmup - 1) // thin
        args['ctrl']['sampling']['iter_save'] = iter_save_wo_warmup + 1 + (warmup - 1) // thin
        refresh = iter // 10 if iter >= 20 else 1
        args['ctrl']['sampling']['refresh'] = args.get('refresh', refresh)
        # NB: argument named "seed" not "random_seed"
        args['random_seed'] = args.get('seed', int(time.time()))

        algorithm = args.get('algorithm', 'NUTS')
        if algorithm == 'HMC':
            args['ctrl']['sampling']['algorithm'] = sampling_algo_t.HMC
        elif algorithm == 'Metropolis':
            args['ctrl']['sampling']['algorithm'] = sampling_algo_t.Metropolis
        elif algorithm == 'NUTS':
            args['ctrl']['sampling']['algorithm'] = sampling_algo_t.NUTS
        else:
            msg = "Invalid value for parameter algorithm (found {}; " \
                "require HMC, Metropolis, or NUTS).".format(algorithm)
            raise ValueError(msg)

        ctrl_lst = args.get('control')
        if ctrl_lst is not None:
            args['ctrl']['sampling']['adapt_engaged'] = ctrl_lst.get("adapt_engaged", True)
            args['ctrl']['sampling']['adapt_gamma'] = ctrl_lst.get("adapt_gamma", 0.05)
            args['ctrl']['sampling']['adapt_delta'] = ctrl_lst.get("adapt_delta", 0.65)
            args['ctrl']['sampling']['adapt_kappa'] = ctrl_lst.get("adapt_kappa", 0.75)
            args['ctrl']['sampling']['adapt_t0'] = ctrl_lst.get("adapt_t0", 10.0)
            args['ctrl']['sampling']['stepsize'] = ctrl_lst.get("stepsize", 1.0)
            args['ctrl']['sampling']['stepsize_jitter'] = ctrl_lst.get("stepsize_jitter", 0.0)

            metric = ctrl_lst.get('metric')
            if metric == "unit_e":
                args['ctrl']['sampling']['metric'] = sampling_metric_t.UNIT_E
            elif metric == "diag_e":
                args['ctrl']['sampling']['metric'] = sampling_metric_t.DIAG_E
            elif metric == "dense_e":
                args['ctrl']['sampling']['metric'] = sampling_metric_t.DENSE_E
            elif metric is None:
                args['ctrl']['sampling']['metric'] = sampling_metric_t.DIAG_E

            if args['ctrl']['sampling']['algorithm'] == sampling_algo_t.NUTS:
                args['ctrl']['sampling']['max_treedepth'] = ctrl_lst.get("max_treedepth", 10)
            elif args['ctrl']['sampling']['algorithm'] == sampling_algo_t.HMC:
                args['ctrl']['sampling']['int_time'] = ctrl_lst.get('int_time', 6.283185307179586476925286766559005768e+00)
            elif args['ctrl']['sampling']['algorithm'] == sampling_algo_t.Metropolis:
                pass
        else:
            args['ctrl']['sampling']['adapt_engaged'] = True
            args['ctrl']['sampling']['adapt_gamma'] = 0.05
            args['ctrl']['sampling']['adapt_delta'] = 0.65
            args['ctrl']['sampling']['adapt_kappa'] = 0.75
            args['ctrl']['sampling']['adapt_t0'] = 10
            args['ctrl']['sampling']['max_treedepth'] = 10
            args['ctrl']['sampling']['metric'] = sampling_metric_t.DIAG_E
            args['ctrl']['sampling']['stepsize'] = 1
            args['ctrl']['sampling']['stepsize_jitter'] = 0
            args['ctrl']['sampling']['int_time'] = 6.283185307179586476925286766559005768e+00

    elif args['method'] == stan_args_method_t.OPTIM:
        args['ctrl'] = args.get('ctrl', dict(optim=dict()))
        args['ctrl']['optim']['iter'] = iter = args.get('iter', 2000)
        algorithm = args.get('algorithm', 'BFGS')
        if algorithm == "BFGS":
            args['ctrl']['optim']['algorithm'] = optim_algo_t.BFGS
        elif algorithm == "Newton":
            args['ctrl']['optim']['algorithm'] = optim_algo_t.Newton
        elif algorithm == "Nesterov":
            args['ctrl']['optim']['algorithm'] = optim_algo_t.Nesterov
        else:
            msg = "Invalid value for parameter algorithm (found {}; " \
                  "require BFGS, Newton, or Nesterov).".format(algorithm)
            raise ValueError(msg)
        refresh = args['ctrl']['optim']['iter'] // 100
        args['ctrl']['optim']['refresh'] = args.get('refresh', refresh)
        if args['ctrl']['optim']['refresh'] < 1:
            args['ctrl']['optim']['refresh'] = 1
        args['ctrl']['optim']['stepsize'] = args.get("stepsize", 1.0)
        args['ctrl']['optim']['init_alpha'] = args.get("init_alpha", 0.001)
        args['ctrl']['optim']['tol_obj'] = args.get("tol_obj", 1e-8)
        args['ctrl']['optim']['tol_grad'] = args.get("tol_grad", 1e-8)
        args['ctrl']['optim']['tol_param'] = args.get("tol_param", 1e-8)
        args['ctrl']['optim']['save_iterations'] = args.get("save_iterations", True)
    elif args['method'] == stan_args_method_t.TEST_GRADIENT:
        pass

    init = args.get('init', "random")
    if isinstance(init, string_types):
        args['init'] = init.encode('ascii')
    elif isinstance(init, Sequence):
        args['init'] = "user".encode('ascii')
        args['init_list'] = init
    else:
        args['init'] = "random".encode('ascii')

    args['init_radius'] = args.get('init_r', 2.0)
    if (args['init_radius'] <= 0):
        args['init'] = "0".encode('ascii')
    # RStan calls validate_args() here
    return args


def _check_seed(seed, raise_exception=True):
    if isinstance(seed, string_types):
        try:
            seed = int(seed)
        except ValueError:
            if raise_exception is True:
                raise ValueError("`seed` must be a string of digits.")
            else:
                logger.warn("`seed` needs to be a string of digits.")
                return None
    elif isinstance(seed, Number):
        seed = int(seed)
    elif seed is None:
        seed = random.randint(0, MAX_UINT)
    else:
        logger.warn('`seed` did not have expected characteristics.')

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
        d[pars[i]] = y.squeeze() if y.shape == (1,) else y
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
        Dictionary uses parameter names as keys. Indexes are column-major order.
        For each parameter there is also a key `par`+'_rowmajor' that stores the
        row-major indexing.

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
            idx = tuple([p])
            return OrderedDict([(par, idx), (par+'_rowmajor', idx)])
        else:
            p = names.index(par)
            idx = starts[p] + np.arange(np.prod(dims[p]))
            idx_rowmajor = starts[p] + _idx_col2rowm(dims[p])
        return OrderedDict([(par, tuple(idx)), (par+'_rowmajor', tuple(idx_rowmajor))])

    indexes = OrderedDict()
    for par in pars:
        indexes.update(par_total_indexes(par))
    return indexes


def _idx_col2rowm(d):
  """Generate indexes to change from col-major to row-major ordering"""
  if 0 == len(d):
      return 1
  if 1 == len(d):
      return np.arange(d[0])
  # order='F' indicates column-major ordering
  idx = np.array(np.arange(np.prod(d))).reshape(d, order='F').T
  return idx.flatten(order='F')


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
    """Get chains for `n`th parameter.

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


def _append_id(file, id, suffix='.csv'):
    fname = os.path.basename(file)
    fpath = os.path.dirname(file)
    fname2 = re.sub(r'\.csv\s*$', '_{}.csv'.format(id), fname)
    if fname2 == fname:
        fname2 = '{}_{}.csv'.format(fname, id)
    return os.path.join(fpath, fname2)


def _writable_sample_file(file, warn=True, wfun=None):
    """Check to see if file is writable, if not use temporary file"""
    if wfun is None:
        wfun = lambda x, y: '"{}" is not writable; use "{}" instead'.format(x, y)
    dir = os.path.dirname(file)
    if os.access(dir, os.W_OK):
        return file
    else:
        dir2 = tempfile.mkdtemp()
        if warn:
            logger.warning(wfun(dir, dir2))
        return os.path.join(dir2, os.path.basename(file))
