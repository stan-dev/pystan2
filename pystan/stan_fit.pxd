# distutils: language = c++
#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

ctypedef unsigned int uint  # needed for templates

cdef extern from "stan_fit.hpp" namespace "pystan":
    ctypedef map[string, pair[vector[double], vector[size_t]]] vars_r_t
    ctypedef map[string, pair[vector[int], vector[size_t]]] vars_i_t
    cdef cppclass stan_fit[M, R]:
        stan_fit(vars_r_t& vars_r, vars_i_t& vars_i) except +
        bool update_param_oi(vector[string] pars)
        vector[double] unconstrain_pars(vars_r_t& vars_r, vars_i_t& vars_i)
        vector[double] constrain_pars(vector[double]& params_r) except +
        double log_prob(vector[double]& par_r) except +
        vector[double] grad_log_prob(vector[double]& par_r) except +
        int num_pars_unconstrained()
        int call_sampler(PyStanArgs&, PyStanHolder&) except +
        vector[string] param_names()
        vector[string] param_names_oi()
        vector[vector[uint]] param_dims()
        vector[string] param_fnames_oi()

    cdef cppclass PyStanArgs:
        bool sample_file_flag
        bool diagnostic_file_flag
        string sample_file
        string diagnostic_file
        int iter
        int warmup
        int thin
        int iter_save
        int iter_save_wo_warmup
        bool save_warmup
        int refresh
        int leapfrog_steps
        double epsilon
        int max_treedepth
        double epsilon_pm
        bool equal_step_sizes
        double delta
        double gamma
        uint random_seed
        string random_seed_src
        uint chain_id
        string chain_id_src
        bool append_samples
        bool test_grad
        bool point_estimate
        bool point_estimate_newton
        string init
        map[string, pair[vector[double], vector[size_t] ] ] init_vars_r
        map[string, pair[vector[int], vector[size_t] ] ] init_vars_i
        string sampler
        bool nondiag_mass

    cdef cppclass PyStanHolder:
        int num_failed
        bool test_grad
        vector[double] inits
        vector[double] par
        double value
        vector[vector[double] ] chains
        vector[string] chain_names
        PyStanArgs args
        vector[double] mean_pars
        double mean_lp__
        string adaptation_info
        vector[vector[double] ] sampler_params
        vector[string] sampler_param_names
