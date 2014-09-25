import cython

import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_kept_samples(int n, dict sim):
    """See documentation in misc.py"""
    cdef int i, j, num_chains, num_iter, ss_index, s_index, num_warmup
    cdef double[:] s, ss
    cdef long[:] perm
    num_chains = sim['chains']
    num_warmup = sim['warmup']
    num_iter = sim['iter']
    nth_key = list(sim['samples'][0]['chains'].keys())[n]
    ss = np.empty((num_iter - num_warmup) * num_chains)
    for i in range(num_chains):
        perm = sim['permutation'][i]
        s = sim['samples'][i]['chains'][nth_key]
        for j in range(num_iter - num_warmup):
            ss_index = i * (num_iter - num_warmup) + j
            s_index = num_warmup + perm[j]
            ss[ss_index] = s[s_index]
    return ss


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_samples(int n, dict sim, inc_warmup):
    """See documentation in misc.py"""
    cdef int i
    cdef double[:] s
    cdef long[:] perm
    cdef int num_chains = sim['chains']
    cdef int num_warmup = sim['warmup']
    cdef int num_iter = sim['iter']
    if num_warmup == 0:
        inc_warmup = True
    nth_key = list(sim['samples'][0]['chains'].keys())[n]
    ss = []
    for i in range(num_chains):
        perm = sim['permutation'][i]
        s = sim['samples'][i]['chains'][nth_key]
        if inc_warmup:
            ss.append(s)
        else:
            ss.append(s[num_warmup:])
    return ss

