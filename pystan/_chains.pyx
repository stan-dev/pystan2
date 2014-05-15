from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.math cimport sqrt

import numpy as np

# Cython doesn't support default templates, so we specify the RNG
cdef extern from "boost/random/additive_combine.hpp" namespace "boost::random":
    cdef cppclass additive_combine_engine[T, U]:
        pass
    ctypedef additive_combine_engine ecuyer1988

cdef extern from "stan/mcmc/chains.hpp" namespace "stan::mcmc":
    cdef cppclass chains[RNG]:
        chains(const vector[string]& param_names)
        void add(const vector[ vector[double] ]& samples)
        int num_params()
        int num_chains()
        double effective_sample_size(const int index)
        double split_potential_scale_reduction(const int index)
        void set_warmup(const int)

cdef _get_samples(dict sim):
    """
    Extract samples into a list of arrays having shape n_samples, n_param

    Parameters
    ----------
    sim : dict
        Contains samples as well as related information.

    Returns
    -------
    samples : list of arrays having shape n_samples, n_params

    """
    # TODO: this overlaps/duplicates with pystan.misc._get_samples
    param_names = sim['fnames_oi']
    sample_dicts = [chain['chains'] for chain in sim['samples']]
    samples = []
    for sample_dict in sample_dicts:
        # check order of parameters
        sample_param_names = [p for p in sample_dict.keys() if p in param_names]
        np.testing.assert_equal(param_names, sample_param_names)
        sample = np.array([sample_dict[param] for param in param_names]).T.tolist()
        samples.append(sample)
    return samples


def effective_sample_size(dict sim, int index):
    """
    Return the effective sample size for the specified parameter
    across all kept samples.

    This implementation matches BDA3's effective size description.

    Current implementation takes the minimum number of samples
    across chains as the number of samples per chain.

    Parameters
    ----------
    sim : dict
        Contains samples as well as related information (warmup, number
        of iterations, etc).
    index : int
        Parameter index

    Returns
    -------
    ess : float
    """
    # convert param names to bytes for C++
    param_names_bytes = [name.encode('ascii') for name in sim['fnames_oi']]
    cdef chains[ecuyer1988] *chainsptr = new chains[ecuyer1988](param_names_bytes)
    if not chainsptr:
        raise MemoryError("Couldn't allocate space for stan::mcmc::chains instance.")
    chainsptr.set_warmup(sim['warmup'])
    samples = _get_samples(sim)
    cdef vector[vector[double]] sample_vector
    for sample in samples:
        sample_vector = sample
        chainsptr.add(sample_vector)
    ess = chainsptr.effective_sample_size(index)
    del chainsptr
    return ess


def split_potential_scale_reduction(dict sim, int index):
    """
    Return the split potential scale reduction (split R hat) for the
    specified parameter.
    
    Current implementation takes the minimum number of samples
    across chains as the number of samples per chain.
    
    Parameters
    ----------
    index : int
        Parameter index

    Returns
    -------
    rhat : float
        Split R hat
    
    """
    # convert param names to bytes for C++
    param_names_bytes = [name.encode('ascii') for name in sim['fnames_oi']]
    cdef chains[ecuyer1988] *chainsptr = new chains[ecuyer1988](param_names_bytes)
    if not chainsptr:
        raise MemoryError("Couldn't allocate space for stan::mcmc::chains instance.")
    chainsptr.set_warmup(sim['warmup'])
    samples = _get_samples(sim)
    cdef vector[vector[double]] sample_vector
    for sample in samples:
        sample_vector = sample
        chainsptr.add(sample_vector)
    rhat = chainsptr.split_potential_scale_reduction(index)
    del chainsptr
    return rhat
