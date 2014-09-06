cimport cython
cimport cython.view
from libc.string cimport memcpy
from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd()
        MatrixXd(int rows, int cols)
        double* data()

# Cython doesn't support default templates, so we specify the RNG explicitly
cdef extern from "boost/random/additive_combine.hpp" namespace "boost::random":
    cdef cppclass additive_combine_engine[T, U]:
        pass
    ctypedef additive_combine_engine ecuyer1988


cdef extern from "stan/mcmc/chains.hpp" namespace "stan::mcmc":
    cdef cppclass chains[RNG]:
        chains(const vector[string]& param_names)
        void add(const vector[ vector[double] ]& samples)
        void add(MatrixXd& sample)
        int num_params()
        int num_chains()
        double effective_sample_size(const int index)
        double split_potential_scale_reduction(const int index)
        void set_warmup(const int)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef MatrixXd _get_sample_matrix(holder, list fnames_oi):
    """
    Extract samples into a MatrixXd of arrays having shape n_iter, n_param

    Parameters
    ----------
    sim : dict
        Contains samples as well as related information.

    fnames_oi : list of str
        Parameter names of interest

    Returns
    -------
    sample : a vector[vector[double]] shape n_iter, n_params

    """
    cdef int i, j, n_col, n_row
    n_col = len(fnames_oi)
    n_row = len(holder['chains'][fnames_oi[0]])
    cdef double[:] chain
    input_arr_cython = cython.view.array(shape=(n_row, n_col), itemsize=sizeof(double), format="d", mode="fortran")
    cdef double[:, :] input_arr = input_arr_cython
    cdef MatrixXd sample = MatrixXd(n_row, n_col)  # column-major

    # convert from row-major (in essence) to column-major
    j = 0
    for par in fnames_oi:
        chain = holder['chains'][par]
        for i in range(n_row):
            input_arr[i, j] = chain[i]
        j = j + 1
    memcpy(sample.data(), &input_arr[0, 0], n_col * n_row * sizeof(double))
    return sample


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
    fnames_oi = sim['fnames_oi']
    for holder in sim['samples']:
        chainsptr.add(_get_sample_matrix(holder, fnames_oi))
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
    fnames_oi = sim['fnames_oi']
    for holder in sim['samples']:
        chainsptr.add(_get_sample_matrix(holder, fnames_oi))
    rhat = chainsptr.split_potential_scale_reduction(index)
    del chainsptr
    return rhat


def effective_sample_size_and_rhat(dict sim, int index):
    """
    Return the effective sample size and rhat for the specified parameter
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
    ess, rhat : tuple of float
    """
    # convert param names to bytes for C++
    param_names_bytes = [name.encode('ascii') for name in sim['fnames_oi']]
    cdef chains[ecuyer1988] *chainsptr = new chains[ecuyer1988](param_names_bytes)
    if not chainsptr:
        raise MemoryError("Couldn't allocate space for stan::mcmc::chains instance.")
    chainsptr.set_warmup(sim['warmup'])
    fnames_oi = sim['fnames_oi']
    for holder in sim['samples']:
        chainsptr.add(_get_sample_matrix(holder, fnames_oi))
    ess = chainsptr.effective_sample_size(index)
    rhat = chainsptr.split_potential_scale_reduction(index)
    del chainsptr
    return (ess, rhat)
