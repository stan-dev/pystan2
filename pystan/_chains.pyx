from libcpp.vector cimport vector
from libc.math cimport sqrt

ctypedef unsigned int uint  # needed for templates

cdef extern from "stan/prob/autocovariance.hpp" namespace "stan::prob":
    void stan_autocovariance "stan::prob::autocovariance<double>"(vector[double]& y, vector[double] acov)

cdef extern from "stan/math.hpp" namespace "stan::math":
    double stan_sum "stan::math::sum"(vector[double]& x)
    double stan_mean "stan::math::mean"(vector[double]& x)
    double stan_variance "stan::math::variance"(vector[double]& x)

cdef double get_chain_mean(dict sim, uint k, uint n):
    allsamples = sim['samples']
    warmup2 = sim['warmup2']

    slst = allsamples[k]['chains']  # chain k, an OrderedDict
    param_names = list(slst.keys())  # e.g., 'beta[1]', 'beta[2]', ...
    cdef vector[double] nv = slst[param_names[n]]  # parameter n
    # Cython will let you slice C++ vectors. Is there a performance hit?
    return stan_mean(nv[warmup2[k]:])


cdef void get_kept_samples(dict sim, uint k, uint n, vector[double]& samples):
    """
    
    Parameters
    ----------
    k : unsigned int
        Chain index
    n : unsigned int
        Parameter index
    """
    cdef uint i
    allsamples = sim['samples']
    n_save = sim['n_save']
    warmup2 = sim['warmup2']

    slst = allsamples[k]['chains']  # chain k, an OrderedDict
    param_names = list(slst.keys())  # e.g., 'beta[1]', 'beta[2]', ...
    cdef vector[double] nv = slst[param_names[n]]  # parameter n
    # NOTE: this creates a copy which is not optimal, RStan avoids this by
    # managing things in C++
    samples.clear()
    for i in range(nv.size() - warmup2[k]):
        samples.push_back(nv[warmup2[k] + i])


cdef vector[double] autocovariance(dict sim, uint k, uint n):
    """
    Returns the autocovariance for the specified parameter in the
    kept samples of the chain specified.
    
    Parameters
    ----------
    k : unsigned int
        Chain index
    n : unsigned int
        Parameter index

    Returns
    -------
    acov : vector[double]

    Note
    ----
    PyStan is profligate with memory here in comparison to RStan. A variety
    of copies are made where RStan passes around references. This is done
    mainly for convenience; the Cython code is simpler.
    """
    cdef vector[double] samples, acov
    get_kept_samples(sim, k, n, samples)
    stan_autocovariance(samples, acov)
    return acov

def effective_sample_size(dict sim, uint n):
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
    n : int
        Parameter index

    Returns
    -------
    ess : int
    """
    cdef uint i, chain
    cdef uint m = sim['chains']

    cdef vector[uint] ns_save = sim['n_save']

    cdef vector[uint] ns_warmup2 = sim['warmup2']

    cdef vector[uint] ns_kept = [s - w for s, w in zip(sim['n_save'], sim['warmup2'])]

    cdef uint n_samples = min(ns_kept)

    cdef vector[vector[double]] acov
    cdef vector[double] acov_chain
    for chain in range(m):
        acov_chain = autocovariance(sim, chain, n)
        acov.push_back(acov_chain)

    cdef vector[double] chain_mean
    cdef vector[double] chain_var
    cdef uint n_kept_samples
    for chain in range(m):
        n_kept_samples = ns_kept[chain]
        chain_mean.push_back(get_chain_mean(sim, chain, n))
        chain_var.push_back(acov[chain][0] * n_kept_samples / (n_kept_samples-1))

    cdef double mean_var = stan_mean(chain_var)
    cdef double var_plus = mean_var * (n_samples-1) / n_samples

    if m > 1:
        var_plus = var_plus + stan_variance(chain_mean)

    cdef vector[double] rho_hat_t
    cdef double rho_hat = 0
    cdef vector[double] acov_t
    cdef uint t = 0
    while t < n_samples and rho_hat >= 0:
        acov_t.clear()
        for chain in range(m):
            acov_t.push_back(acov[chain][t])
        rho_hat = 1 - (mean_var - stan_mean(acov_t)) / var_plus
        if rho_hat >= 0:
            rho_hat_t.push_back(rho_hat)
        t += 1

    cdef double ess = m * n_samples
    if rho_hat_t.size() > 0:
        ess = ess / (1 + 2 * stan_sum(rho_hat_t))

    return ess


def split_potential_scale_reduction(dict sim, uint n):
    """
    Return the split potential scale reduction (split R hat) for the
    specified parameter.
    
    Current implementation takes the minimum number of samples
    across chains as the number of samples per chain.
    
    Parameters
    ----------
    n : unsigned int
        Parameter index

    Returns
    -------
    rhat : float
        Split R hat
    
    """
    cdef uint i, chain
    cdef uint n_chains = sim['chains']

    cdef vector[uint] ns_save = sim['n_save']

    cdef vector[uint] ns_warmup2 = sim['warmup2']

    cdef vector[uint] ns_kept = [s - w for s, w in zip(sim['n_save'], sim['warmup2'])]

    cdef uint n_samples = min(ns_kept)

    if n_samples % 2 == 1:
        n_samples = n_samples - 1

    cdef vector[double] split_chain_mean, split_chain_var
    cdef vector[double] samples, split_chain
    for chain in range(n_chains):
        samples.clear()
        get_kept_samples(sim, chain, n, samples)
        # c++ vector assign isn't available in Cython; this is a workaround
        split_chain.clear()
        for i in range(n_samples/2):
            split_chain.push_back(samples[i])
        split_chain_mean.push_back(stan_mean(split_chain))
        split_chain_var.push_back(stan_variance(split_chain))

        split_chain.clear()
        for i in range(n_samples/2, n_samples):
            split_chain.push_back(samples[i])
        split_chain_mean.push_back(stan_mean(split_chain))
        split_chain_var.push_back(stan_variance(split_chain))

    cdef double var_between = n_samples/2 * stan_variance(split_chain_mean)
    cdef double var_within = stan_mean(split_chain_var)

    cdef double srhat = sqrt((var_between/var_within + n_samples/2 -1)/(n_samples/2))
    return srhat

def stan_prob_autocovariance(vector[double] dv):
    cdef vector[double] acov
    stan_autocovariance(dv, acov)
    return acov
