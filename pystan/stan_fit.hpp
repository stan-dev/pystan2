#ifndef __PYSTAN__STAN_FIT_HPP__
#define __PYSTAN__STAN_FIT_HPP__

#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <stan/version.hpp>
// #include <stan/io/cmd_line.hpp>
// #include <stan/io/dump.hpp>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/agrad/agrad.hpp>

#include <stan/model/util.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/nesterov_gradient.hpp>
#include <stan/optimization/bfgs.hpp>

#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include "py_var_context.hpp"
#include "mcmc_output.hpp"

// REF: stan/gm/command.hpp 
// REF: rstan/rstan/inst/include/rstan/stan_fit.hpp

typedef std::map<std::string, std::pair<std::vector<double>, std::vector<size_t> > > vars_r_t;
typedef std::map<std::string, std::pair<std::vector<int>, std::vector<size_t> > > vars_i_t;

namespace pystan {

  /* simple class to store arguments provided by Python */
  class PyStanArgs {
      
    public:
      bool sample_file_flag;
      bool diagnostic_file_flag;
      std::string sample_file;  /* file for outputting the samples */
      std::string diagnostic_file;
      int iter;  /* number of iterations */
      int warmup;  /* number of warmup */
      int thin;
      int iter_save;  /* number of iterations saved */
      int iter_save_wo_warmup;  /* number of iterations saved without warmup */
      bool save_warmup;  /* weather to save warmup samples (always true now) */
      int refresh;
      int leapfrog_steps;
      double epsilon;
      int max_treedepth;
      double epsilon_pm;
      bool equal_step_sizes;
      double delta;
      double gamma;
      unsigned int random_seed;
      std::string random_seed_src;  /* "user" or "default" */
      unsigned int chain_id;
      std::string chain_id_src;  /* "user" or "default" */
      bool append_samples;
      bool test_grad;
      int point_estimate;
      std::string init;
      /* FIXME: size_t may be for R's benefit, unnecessary here? */
      vars_r_t init_vars_r;
      vars_i_t init_vars_i;
      std::string sampler; /* HMC, NUTS1, NUTS2 */
      bool nondiag_mass;
  };
  
  /* simple class to store data for Python */
  class PyStanHolder {
  
      public:
          int num_failed;
          bool test_grad;
          std::vector<double> inits;
          std::vector<double> par;
          double value;
          std::vector<std::vector<double> > chains;
          std::vector<std::string> chain_names;
          PyStanArgs args;
          std::vector<double> mean_pars;
          double mean_lp__;
          std::string adaptation_info;
          std::vector<std::vector<double> > sampler_params;
          std::vector<std::string> sampler_param_names;
  };


    /* extra functions from RStan's stan_args.hpp */

    void write_comment(std::ostream& o) {
      o << "#" << std::endl;
    }
  
    template <typename M>
    void write_comment(std::ostream& o,
                       const M& msg) {
      o << "# " << msg << std::endl;
    }
  
    template <typename K, typename V>
    void write_comment_property(std::ostream& o,
                                const K& key,
                                const V& val) {
      o << "# " << key << "=" << val << std::endl;
    }

    /** 
     * Find the index of an element in a vector. 
     * @param v the vector in which an element are searched. 
     * @param e the element that we are looking for. 
     * @return If e is in v, return the index (0 to size - 1);
     *  otherwise, return the size. 
     */
   
    template <class T>
    size_t find_index(const std::vector<T>& v, const T& e) {
      return std::distance(v.begin(), std::find(v.begin(), v.end(), e));  
    } 

    /**
     *@tparam T The type by which we use for dimensions. T could be say size_t
     * or unsigned int. This whole business (not using size_t) is due to that
     * Rcpp::wrap/as does not support size_t on some platforms and R could not
     * deal with 64bits integers. 
     *
     */ 
    template <class T> 
    size_t calc_num_params(const std::vector<T>& dim) {
      T num_params = 1;
      for (size_t i = 0;  i < dim.size(); ++i)
        num_params *= dim[i];
      return num_params;
    }

    template <class T> 
    void calc_starts(const std::vector<std::vector<T> >& dims,
                     std::vector<T>& starts) { 
      starts.resize(0); 
      starts.push_back(0); 
      for (size_t i = 1; i < dims.size(); ++i)
        starts.push_back(starts[i - 1] + calc_num_params(dims[i - 1]));
    }

    template <class T> 
    T calc_total_num_params(const std::vector<std::vector<T> >& dims) {
      T num_params = 0;
      for (size_t i = 0; i < dims.size(); ++i)
        num_params += calc_num_params(dims[i]);
      return num_params;
    }

    /**
     *  Get the parameter indexes for a vector(array) parameter.
     *  For example, we have parameter beta, which has 
     *  dimension [2,3]. Then this function gets 
     *  the indexes as (if col_major = false)
     *  [0,0], [0,1], [0,2] 
     *  [1,0], [1,1], [1,2] 
     *  or (if col_major = true) 
     *  [0,0], [1,0] 
     *  [0,1], [1,1] 
     *  [0,2], [121] 
     *
     *  @param dim[in] the dimension of parameter
     *  @param idx[out] for keeping all the indexes
     *
     *  <p> when idx is empty (size = 0), idx 
     *  would contains an empty vector. 
     * 
     *
     */
    
    template <class T>
    void expand_indices(std::vector<T> dim,
                        std::vector<std::vector<T> >& idx,
                        bool col_major = false) {
      size_t len = dim.size();
      idx.resize(0);
      size_t total = calc_num_params(dim);
      std::vector<size_t> loopj;
      for (size_t i = 1; i <= len; ++i)
        loopj.push_back(len - i);
    
      if (col_major)
        for (size_t i = 0; i < len; ++i)
          loopj[i] = len - 1 - loopj[i];
    
      idx.push_back(std::vector<T>(len, 0));
      for (size_t i = 1; i < total; i++) {
        std::vector<T>  v(idx.back());
        for (size_t j = 0; j < len; ++j) {
          size_t k = loopj[j];
          if (v[k] < dim[k] - 1) {
            v[k] += 1;
            break;
          }
          v[k] = 0;
        }
        idx.push_back(v);
      }
    }

    /**
     * Get the names for an array of given dimensions 
     * in the way of column majored. 
     * For example, if we know an array named `a`, with
     * dimensions of [2, 3, 4], the names then are (starting
     * from 0):
     * a[0,0,0]
     * a[1,0,0]
     * a[0,1,0]
     * a[1,1,0]
     * a[0,2,0]
     * a[1,2,0]
     * a[0,0,1]
     * a[1,0,1]
     * a[0,1,1]
     * a[1,1,1]
     * a[0,2,1]
     * a[1,2,1]
     * a[0,0,2]
     * a[1,0,2]
     * a[0,1,2]
     * a[1,1,2]
     * a[0,2,2]
     * a[1,2,2]
     * a[0,0,3]
     * a[1,0,3]
     * a[0,1,3]
     * a[1,1,3]
     * a[0,2,3]
     * a[1,2,3]
     *
     * @param name The name of the array variable 
     * @param dim The dimensions of the array 
     * @param fnames[out] Where the names would be pushed. 
     * @param first_is_one[true] Where to start for the first index: 0 or 1. 
     *
     */
    template <class T> void
    get_flatnames(const std::string& name,
                  const std::vector<T>& dim,
                  std::vector<std::string>& fnames,
                  bool col_major = true,
                  bool first_is_one = true) {

      fnames.clear(); 
      if (0 == dim.size()) {
        fnames.push_back(name);
        return;
      }

      std::vector<std::vector<T> > idx;
      expand_indices(dim, idx, col_major); 
      size_t first = first_is_one ? 1 : 0;
      for (typename std::vector<std::vector<T> >::const_iterator it = idx.begin();
           it != idx.end();
           ++it) {
        std::stringstream stri;
        stri << name << "[";

        size_t lenm1 = it -> size() - 1;
        for (size_t i = 0; i < lenm1; i++)
          stri << ((*it)[i] + first) << ",";
        stri << ((*it)[lenm1] + first) << "]";
        fnames.push_back(stri.str());
      }
    }

    // vectorize get_flatnames 
    template <class T> 
    void get_all_flatnames(const std::vector<std::string>& names, 
                           const std::vector<T>& dims, 
                           std::vector<std::string>& fnames, 
                           bool col_major = true) {
      fnames.clear(); 
      for (size_t i = 0; i < names.size(); ++i) {
        std::vector<std::string> i_names; 
        get_flatnames(names[i], dims[i], i_names, col_major, false); // col_major = true, first_is_one = false for PyStan (true for RStan)
        fnames.insert(fnames.end(), i_names.begin(), i_names.end());
      } 
    } 

    /* To facilitate transform an array variable ordered by col-major index
     * to row-major index order by providing the transforming indices.
     * For example, we have "x[2,3]", then if ordered by col-major, we have
     * 
     * x[1,1], x[2,1], x[1,2], x[2,2], x[1,3], x[3,1]
     * 
     * Then the indices for transforming to row-major order are 
     * [0, 2, 4, 1, 3, 5] + start. 
     *
     * @param dim[in] the dimension of the array variable, empty means a scalar
     * @param midx[out] store the indices for mapping col-major to row-major
     * @param start shifts the indices with a starting point
     *
     */ 
    template <typename T, typename T2>
    void get_indices_col2row(const std::vector<T>& dim, std::vector<T2>& midx,
                             T start = 0) {
      size_t len = dim.size();
      if (len < 1) { 
        midx.push_back(start); 
        return; 
      }
    
      std::vector<T> z(len, 1);
      for (size_t i = 1; i < len; i++) {
        z[i] *= z[i - 1] * dim[i - 1];
      } 
    
      T total = calc_num_params(dim);
      midx.resize(total);
      std::fill_n(midx.begin(), total, start);
      std::vector<T> v(len, 0);
      for (T i = 1; i < total; i++) {
        for (size_t j = 0; j < len; ++j) {
          size_t k = len - j - 1;
          if (v[k] < dim[k] - 1) {
            v[k] += 1;
            break; 
          }
          v[k] = 0; 
        } 
        // v is the index of the ith element by row-major, for example v=[0,1,2]. 
        // obtain the position for v if it is col-major indexed. 
        T pos = 0;
        for (size_t j = 0; j < len; j++) 
          pos += z[j] * v[j];
        midx[i] += pos;
      } 
    } 
   
    template <class T>
    void get_all_indices_col2row(const std::vector<std::vector<T> >& dims,
                                 std::vector<size_t>& midx) {
      midx.clear();
      std::vector<T> starts; 
      calc_starts(dims, starts);
      for (size_t i = 0; i < dims.size(); ++i) {
        std::vector<size_t> midxi;
        get_indices_col2row(dims[i], midxi, starts[i]);
        midx.insert(midx.end(), midxi.begin(), midxi.end());
      } 
    } 

    bool do_print(int n, int refresh, int last = 0) {
      if (refresh < 1) return false;
      return (n == 0) || ((n + 1) % refresh == 0) || (n == last);
    }

    void print_progress(int m, int finish, int refresh, bool warmup) {
      int it_print_width = std::ceil(std::log10(finish));
      if (do_print(m, refresh, finish - 1)) {
        std::cout << "\rIteration: ";
        std::cout << std::setw(it_print_width) << (m + 1)
                         << " / " << finish;
        std::cout << " [" << std::setw(3) 
                         << static_cast<int>((100.0 * (m + 1)) / finish)
                         << "%] ";
        std::cout << (warmup ? " (Warmup)" : " (Sampling)");
      }
    }

    template <class Model>
    std::vector<std::string> get_param_names(Model& m) { 
      std::vector<std::string> names;
      m.get_param_names(names);
      names.push_back("lp__");
      return names; 
    }

    template <class T>
    void print_vector(const std::vector<T>& v, std::ostream& o, 
                      const std::vector<size_t>& midx, 
                      const std::string& sep = ",") {
      if (v.size() > 0)
        o << v[0];
      for (size_t i = 1; i < v.size(); i++)
        o << sep << v[midx.at(i)];
      o << std::endl;
    }

    template <class T>
    void print_vector(const std::vector<T>& v, std::ostream& o, 
                      const std::string& sep = ",") {
      if (v.size() > 0)
        o << v[0];
      for (size_t i = 1; i < v.size(); i++)
        o << sep << v[i];
      o << std::endl;
    }

    template <class Sampler, class Model, class RNG>
    void run_markov_chain(Sampler& sampler,
                          int num_warmup, int num_iterations,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          mcmc_output<Model> outputer, 
                          stan::mcmc::sample& init_s,
                          Model& model,
                          std::vector<std::vector<double> >& chains, 
                          int& iter_save_i,
                          const std::vector<size_t>& qoi_idx,
                          std::vector<double>& sum_pars,
                          double& sum_lp,
                          std::vector<std::vector<double> >& sampler_params, 
                          std::vector<std::vector<double> >& iter_params,
                          std::string& adaptation_info, 
                          RNG& base_rng) {
      int start = 0;
      int end = num_warmup; 
      if (!warmup) { 
        start = num_warmup;
        end = num_iterations;
      } 
      for (int m = start; m < end; ++m) {
        print_progress(m, num_iterations, refresh, warmup);
        // FIXME: Python equivalent? R_CheckUserInterrupt(); 
        init_s = sampler.transition(init_s);
        if (save && (((m - start) % num_thin) == 0)) {
          outputer.output_sample_params(base_rng, init_s, sampler, model,
                                        chains, warmup,
                                        sampler_params, iter_params,
                                        sum_pars, sum_lp, qoi_idx,
                                        iter_save_i, &std::cout);
          iter_save_i++;
          outputer.output_diagnostic_params(init_s, sampler);
        }
      }
    }

    template <class Sampler, class Model, class RNG>
    void warmup_phase(Sampler& sampler,
                      int num_warmup,
                      int num_iterations,
                      int num_thin,
                      int refresh,
                      bool save,
                      mcmc_output<Model> outputer,
                      stan::mcmc::sample& init_s,
                      Model& model,
                      std::vector<std::vector<double> >& chains, 
                      int& iter_save_i,
                      const std::vector<size_t>& qoi_idx,
                      std::vector<double>& sum_pars,
                      double& sum_lp,
                      std::vector<std::vector<double> >& sampler_params,
                      std::vector<std::vector<double> >& iter_params,
                      std::string& adaptation_info, 
                      RNG& base_rng) {
      run_markov_chain<Sampler, Model, RNG>(sampler, num_warmup, num_iterations, 
                                            num_thin,
                                            refresh, save, true,
                                            outputer,
                                            init_s, model, chains, iter_save_i, qoi_idx,
                                            sum_pars, sum_lp, sampler_params, iter_params,
                                            adaptation_info, base_rng);
    }

    template <class Sampler, class Model, class RNG>
    void sample_phase(Sampler& sampler,
                      int num_warmup,
                      int num_iterations,
                      int num_thin,
                      int refresh,
                      bool save,
                      mcmc_output<Model> outputer,
                      stan::mcmc::sample& init_s,
                      Model& model,
                      std::vector<std::vector<double> >& chains, 
                      int& iter_save_i,
                      const std::vector<size_t>& qoi_idx,
                      std::vector<double>& sum_pars,
                      double& sum_lp,
                      std::vector<std::vector<double> >& sampler_params,
                      std::vector<std::vector<double> >& iter_params,
                      std::string& adaptation_info, 
                      RNG& base_rng) {
      run_markov_chain<Sampler, Model, RNG>(sampler, num_warmup, num_iterations, num_thin,
                                            refresh, save, false,
                                            outputer,
                                            init_s, model, chains, iter_save_i, qoi_idx,
                                            sum_pars, sum_lp, sampler_params, iter_params,
                                            adaptation_info,
                                            base_rng);
    }
    

    /**
     * Cast a size_t vector to an unsigned int vector. 
     * The reason is that first Rcpp::wrap/as does not 
     * support size_t on some platforms; second R 
     * could not deal with 64bits integers.  
     */ 

    std::vector<unsigned int> 
    sizet_to_uint(std::vector<size_t> v1) {
      std::vector<unsigned int> v2(v1.size());
      for (size_t i = 0; i < v1.size(); ++i) 
        v2[i] = static_cast<unsigned int>(v1[i]);
      return v2;
    } 

    template <class Model>
    std::vector<std::vector<unsigned int> > get_param_dims(Model& m) {
      std::vector<std::vector<size_t> > dims; 
      m.get_dims(dims); 

      std::vector<std::vector<unsigned int> > uintdims; 
      for (std::vector<std::vector<size_t> >::const_iterator it = dims.begin();
           it != dims.end(); 
           ++it) 
        uintdims.push_back(sizet_to_uint(*it)); 

      std::vector<unsigned int> scalar_dim; // for lp__
      uintdims.push_back(scalar_dim); 
      return uintdims; 
    } 


    /**
     * @tparam Model 
     * @tparam RNG 
     *
     * @param args: the instance that wraps the arguments passed for sampling.
     * @param model: the model instance.
     * @param holder[out] the object to hold all the information returned to Python.
     * @param qoi_idx: the indexes for all parameters of interest.  
     * @param fnames_oi: the parameter names of interest.  
     * @param base_rng: the boost RNG instance. 
     */
    template <class Model, class RNG> 
    int sampler_command(PyStanArgs& args, Model& model,
                        PyStanHolder& holder,
                        const std::vector<size_t>& qoi_idx, 
                        const std::vector<std::string>& fnames_oi, RNG& base_rng) {
      bool sample_file_flag = args.sample_file_flag;
      bool diagnostic_file_flag = args.diagnostic_file_flag;
      std::string sample_file = args.sample_file;
      std::string diagnostic_file = args.diagnostic_file;
      int num_iterations = args.iter;
      int num_warmup = args.warmup;
      int num_thin = args.thin; 
      int iter_save = args.iter_save;
      int iter_save_wo_warmup = args.iter_save_wo_warmup;
      int leapfrog_steps = args.leapfrog_steps;
      unsigned int random_seed = args.random_seed;
      double epsilon = args.epsilon;
      // bool epsilon_adapt = (epsilon <= 0.0);
      bool equal_step_sizes = args.equal_step_sizes;
      int max_treedepth = args.max_treedepth;
      double epsilon_pm = args.epsilon_pm;
      double delta = args.delta;
      double gamma = args.gamma;
      int refresh = args.refresh;
      unsigned int chain_id = args.chain_id;
      bool test_grad = args.test_grad;
      int point_estimate = args.point_estimate;
      bool nondiag_mass = args.nondiag_mass;
      bool save_warmup = args.save_warmup;

      base_rng.seed(random_seed);
      // (2**50 = 1T samples, 1000 chains)
      static boost::uintmax_t DISCARD_STRIDE = 
        static_cast<boost::uintmax_t>(1) << 50;
      // std::cout << "DISCARD_STRIDE=" << DISCARD_STRIDE << std::endl;
      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      
      std::vector<double> cont_params;
      std::vector<int> disc_params;
      std::string init_val = args.init;
      int num_init_tries = 0;
      // parameter initialization
      if (init_val == "0") {
        disc_params = std::vector<int>(model.num_params_i(),0);
        cont_params = std::vector<double>(model.num_params_r(),0.0);
        double init_log_prob;
        std::vector<double> init_grad;
        try {
          init_log_prob
            = stan::model::log_prob_grad<true,true>(model,
                                                    cont_params, 
                                                    disc_params, 
                                                    init_grad, 
                                                    &std::cout);
        } catch (const std::domain_error& e) {
          std::string msg("Domain error during initialization with 0:\n"); 
          msg += e.what();
          throw std::runtime_error(msg);
        }
        if (!boost::math::isfinite(init_log_prob))  
          throw std::runtime_error("Error during initialization with 0: vanishing density.");
        for (size_t i = 0; i < init_grad.size(); i++) {
          if (!boost::math::isfinite(init_grad[i])) 
            throw std::runtime_error("Error during initialization with 0: divergent gradient.");
        }
      } else if (init_val == "user") {
        try { 
          pystan::io::py_var_context init_var_context(args.init_vars_r, args.init_vars_i);
          model.transform_inits(init_var_context,disc_params,cont_params);
        } catch (const std::exception& e) {
          std::string msg("Error during user-specified initialization:\n"); 
          msg += e.what(); 
          throw std::runtime_error(msg);
        } 
      } else {
        init_val = "random"; 
        // init_rng generates uniformly from -2 to 2
        boost::random::uniform_real_distribution<double> 
          init_range_distribution(-2.0,2.0);
        boost::variate_generator<RNG&, boost::random::uniform_real_distribution<double> >
          init_rng(base_rng,init_range_distribution);

        disc_params = std::vector<int>(model.num_params_i(),0);
        cont_params = std::vector<double>(model.num_params_r());

        // retry inits until get a finite log prob value
        std::vector<double> init_grad;
        static int MAX_INIT_TRIES = 100;
        for (; num_init_tries < MAX_INIT_TRIES; ++num_init_tries) {
          for (size_t i = 0; i < cont_params.size(); ++i)
            cont_params[i] = init_rng();
          double init_log_prob;
          try {
            init_log_prob 
              = stan::model::log_prob_grad<true,true>(model,cont_params,disc_params,init_grad,&std::cout);
          } catch (const std::domain_error& e) {
            // write_error_msg(&std::cout, e);
            std::cout << e.what(); 
            std::cout << "Rejecting proposed initial value with zero density." << std::endl;
            continue;
          } 
          if (!boost::math::isfinite(init_log_prob))
            continue;
          for (size_t i = 0; i < init_grad.size(); ++i)
            if (!boost::math::isfinite(init_grad[i]))
              continue;
          break;
        }
        if (num_init_tries == MAX_INIT_TRIES) {
            std::cout << "Initialization failed after " << MAX_INIT_TRIES 
                           << " attempts. "
                           << " Try specifying initial values,"
                           << " reducing ranges of constrained values,"
                           << " or reparameterizing the model."
                           << std::endl;
          return -1;
        }
      }
      // keep a record of the initial values 
      std::vector<double> initv; 
      model.write_array(base_rng, cont_params,disc_params,initv); 

      if (test_grad) {
        std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;
        std::stringstream ss; 
        int num_failed = stan::model::test_gradients<true,true>(model,cont_params,disc_params,1e-6,1e-6,ss);
        std::cout << ss.str() << std::endl; 
        holder.num_failed = num_failed; 
        holder.test_grad = true;
        holder.inits = initv; 
        return 0;
      } 

      std::fstream sample_stream; 
      std::fstream diagnostic_stream;
      bool append_samples(args.append_samples);
      if (sample_file_flag) {
        std::ios_base::openmode samples_append_mode
          = append_samples ? (std::fstream::out | std::fstream::app)
                           : std::fstream::out;
        sample_stream.open(sample_file.c_str(), samples_append_mode);
      }

      if (point_estimate == 3) { // bfgs
        std::cout << "STAN OPTIMIZATION COMMAND" << std::endl;
        std::cout << "init = " << init_val << std::endl;
        if (num_init_tries > 0)
          std::cout << "init tries = " << num_init_tries << std::endl;
        if (sample_file_flag) 
          std::cout << "output = " << sample_file << std::endl;
        std::cout << "save_warmup = " << save_warmup << std::endl;
        std::cout << "epsilon = " << epsilon << std::endl;
        std::cout << "seed = " << random_seed << std::endl;
        
        if (sample_file_flag) { 
          write_comment(sample_stream,"Point Estimate Generated by Stan");
          write_comment(sample_stream);
          write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
          write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
          write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
          write_comment_property(sample_stream,"init",init_val);
          write_comment_property(sample_stream,"save_warmup",save_warmup);
          write_comment_property(sample_stream,"seed",random_seed);
          write_comment_property(sample_stream,"epsilon",epsilon);
          write_comment(sample_stream);
          
          sample_stream << "lp__,"; // log probability first
          model.write_csv_header(sample_stream);
        } 
        
        stan::optimization::BFGSLineSearch<Model> ng(model, cont_params, disc_params,
                                                     &std::cout);
        if (epsilon > 0)
          ng._opts.alpha0 = epsilon;
        
        double lp = ng.logp();
        
        std::cout << "initial log joint probability = " << lp << std::endl;
        int m = 0;
        int ret = 0;
        for (int i = 0; i < num_iterations && ret == 0; i++) {
          ret = ng.step();
          lp = ng.logp();
          ng.params_r(cont_params);
          if (do_print(i, 50*refresh)) {
            std::cout << "    Iter ";
            std::cout << "     log prob ";
            std::cout << "       ||dx|| ";
            std::cout << "     ||grad|| ";
            std::cout << "      alpha ";
            std::cout << "     alpha0 ";
            std::cout << " # evals ";
            std::cout << " Notes " << std::endl;
          }
          if (do_print(i, refresh) || ret != 0 || !ng.note().empty()) {
            std::cout << " " << std::setw(7) << (m + 1) << " ";
            std::cout << " " << std::setw(12) << std::setprecision(6) << lp << " ";
            std::cout << " " << std::setw(12) << std::setprecision(6) << ng.prev_step_size() << " ";
            std::cout << " " << std::setw(12) << std::setprecision(6) << ng.curr_g().norm() << " ";
            std::cout << " " << std::setw(10) << std::setprecision(4) << ng.alpha() << " ";
            std::cout << " " << std::setw(10) << std::setprecision(4) << ng.alpha0() << " ";
            std::cout << " " << std::setw(7) << ng.grad_evals() << " ";
            std::cout << " " << ng.note() << " ";
            std::cout << std::endl;
            std::cout.flush();
          }
          m++;
          if (save_warmup) {
            sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,sample_stream);
            sample_stream.flush();
          }
        }
        if (ret != 0)
          std::cout << "Optimization terminated with code " << ret << std::endl;
        else 
          std::cout << "Maximum number of iterations hit, optimization terminated." 
                           << std::endl;
        
        std::vector<double> params_inr_etc; // cont, disc, and others
        model.write_array(base_rng,cont_params,disc_params,params_inr_etc);
        holder.par = params_inr_etc; 
        holder.value = lp;
        if (sample_file_flag) { 
          sample_stream << lp << ',';
          print_vector(params_inr_etc, sample_stream);
          sample_stream.close();
        }
        return 0;
      }

      if (point_estimate == 1) { //Newton's method 
        std::cout << "STAN OPTIMIZATION COMMAND" << std::endl;
        if (sample_file_flag) {
          write_comment(sample_stream,"Point Estimate Generated by Stan");
          write_comment(sample_stream);
          write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
          write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
          write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
          write_comment_property(sample_stream,"init",init_val);
          write_comment_property(sample_stream,"seed",random_seed);
          write_comment(sample_stream);

          sample_stream << "lp__,"; // log probability first
          model.write_csv_header(sample_stream);
        }
        std::vector<double> gradient;
        double lp = stan::model::log_prob_grad<true,true>(model, cont_params, disc_params, gradient);
        
        double lastlp = lp - 1;
        std::cout << "initial log joint probability = " << lp << std::endl;
        int m = 0;
        while ((lp - lastlp) / fabs(lp) > 1e-8) {
          // FIXME: Python equivalent? R_CheckUserInterrupt(); 
          lastlp = lp;
          lp = stan::optimization::newton_step(model, cont_params, disc_params);
          std::cout << "Iteration ";
          std::cout << std::setw(2) << (m + 1) << ". ";
          std::cout << "Log joint probability = " << std::setw(10) << lp;
          std::cout << ". Improved by " << (lp - lastlp) << ".";
          std::cout << std::endl;
          std::cout.flush();
          m++;
          if (sample_file_flag) { 
            sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,sample_stream);
          }
        }
        std::vector<double> params_inr_etc;
        model.write_array(base_rng, cont_params, disc_params, params_inr_etc);
        holder.par = params_inr_etc; 
        holder.value = lp;
        // holder.attr("point_estimate") = Rcpp::wrap(true); 

        if (sample_file_flag) { 
          sample_stream << lp << ',';
          print_vector(params_inr_etc, sample_stream);
          sample_stream.close();
        }
        return 0;
      } 

      if (point_estimate == 2) { // nesterov's accelerated gradient method 
        if (sample_file_flag) {
          write_comment(sample_stream,"Point Estimate Generated by Stan");
          write_comment(sample_stream);
          write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
          write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
          write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
          write_comment_property(sample_stream,"init",init_val);
          write_comment_property(sample_stream,"seed",random_seed);
          write_comment(sample_stream);

          sample_stream << "lp__,"; // log probability first
          model.write_csv_header(sample_stream);
        }

        stan::optimization::NesterovGradient<Model> ng(model, cont_params, disc_params,
                                                       -1.0,&std::cout);

        double lp = ng.logp();
        double lastlp = lp - 1;
        std::cout << "initial log joint probability = " << lp << std::endl;
        int m = 0;
        for (int i = 0; i < num_iterations; i++) {
          // FIXME: Python equivalent? R_CheckUserInterrupt(); 
          lastlp = lp;
          lp = ng.step();
          ng.params_r(cont_params);
          if (do_print(i, refresh)) {
            std::cout << "Iteration ";
            std::cout << std::setw(2) << (m + 1) << ". ";
            std::cout << "Log joint probability = " << std::setw(10) << lp;
            std::cout << ". Improved by " << (lp - lastlp) << ".";
            std::cout << std::endl;
            std::cout.flush();
          }
          // FIXME: not working -- seems to fail at iteration 20
          m++;
          if (sample_file_flag) {
            sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,sample_stream);
          }
        }

        std::vector<double> params_inr_etc; // continuous, discrete, and others 
        sample_stream << lp << ',';
        model.write_array(base_rng,cont_params,disc_params,params_inr_etc);
        holder.par = params_inr_etc; 
        holder.value = lp;
        if (sample_file_flag) { 
          sample_stream << lp << ',';
          print_vector(params_inr_etc, sample_stream);
          sample_stream.close();
        }
        return 0;
      } 

      if (diagnostic_file_flag)
        diagnostic_stream.open(diagnostic_file.c_str(), std::fstream::out);

      mcmc_output<Model> outputer(&sample_stream, &diagnostic_stream);
      
      std::vector<std::vector<double> > chains; 
      std::vector<double> mean_pars;
      mean_pars.resize(initv.size(), 0);
      double mean_lp(0);
      std::vector<std::vector<double> > sampler_params;
      std::vector<std::vector<double> > iter_params;
      std::vector<std::string> sampler_param_names;
      std::vector<std::string> iter_param_names;
      std::string adaptation_info;

      for (unsigned int i = 0; i < qoi_idx.size(); i++) 
        chains.push_back(std::vector<double>(iter_save, 0)); 
      if (sample_file_flag) {
        write_comment(sample_stream,"Samples Generated by Stan");
        write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
        write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
        write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
        // FIXME: to implement
        //args.write_args_as_comment(sample_stream); 
      } 
      if (diagnostic_file_flag) {
        write_comment(diagnostic_stream,"Samples Generated by Stan");
        write_comment_property(diagnostic_stream,"stan_version_major",stan::MAJOR_VERSION);
        write_comment_property(diagnostic_stream,"stan_version_minor",stan::MINOR_VERSION);
        write_comment_property(diagnostic_stream,"stan_version_patch",stan::PATCH_VERSION);
        //args.write_args_as_comment(diagnostic_stream);
      } 
     
      double warmDeltaT;
      double sampleDeltaT;
      int iter_save_i = 0;
      if (nondiag_mass) { 
        args.sampler = "NUTS(nondiag)";
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_dense_e_nuts<Model, RNG> a_Dm_nuts;
        a_Dm_nuts sampler(model, base_rng, num_warmup);
        outputer.set_output_names(s, sampler, model, iter_param_names, sampler_param_names);
        outputer.init_sampler_params(sampler_params, iter_save);
        outputer.init_iter_params(iter_params, iter_save);

        if (!append_samples) {
          outputer.print_sample_names();
          outputer.output_diagnostic_names(s, sampler, model);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_max_depth(max_treedepth);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_Dm_nuts, Model, RNG>(sampler, num_warmup, num_iterations, num_thin, 
                                            refresh, save_warmup, 
                                            outputer, 
                                            s, model, chains, iter_save_i,
                                            qoi_idx, mean_pars, mean_lp,
                                            sampler_params, iter_params, adaptation_info,
                                            base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        outputer.output_adapt_finish(sampler, adaptation_info);
        // Sampling
        start = clock();
        sample_phase<a_Dm_nuts, Model, RNG>(sampler, num_warmup, num_iterations, num_thin, 
                                            refresh, true, 
                                            outputer,
                                            s, model, chains, iter_save_i,
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, iter_params, adaptation_info,  
                                            base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } else if (leapfrog_steps < 0 && !equal_step_sizes) { 
        // Euclidean NUTS with Diagonal Metric
        args.sampler = "NUTS2";
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_diag_e_nuts<Model, RNG> a_dm_nuts;
        a_dm_nuts sampler(model, base_rng, num_warmup);
        outputer.set_output_names(s, sampler, model, iter_param_names, sampler_param_names);
        outputer.init_sampler_params(sampler_params, iter_save);
        outputer.init_iter_params(iter_params, iter_save);
        if (!append_samples) {
          outputer.print_sample_names();
          outputer.output_diagnostic_names(s, sampler, model);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_max_depth(max_treedepth);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_dm_nuts, Model, RNG>(sampler, num_warmup, num_iterations, num_thin,
                                            refresh, save_warmup, 
                                            outputer,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, iter_params, adaptation_info,  
                                            base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        outputer.output_adapt_finish(sampler, adaptation_info); 
        
        // Sampling
        start = clock();
        sample_phase<a_dm_nuts, Model, RNG>(sampler, num_warmup, num_iterations,
                                            num_thin, refresh, true, 
                                            outputer,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, iter_params, adaptation_info,
                                            base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } else if (leapfrog_steps < 0 && equal_step_sizes) {
        // Euclidean NUTS with Unit Metric
        args.sampler = "NUTS1";
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_unit_e_nuts<Model, RNG> a_um_nuts;
        a_um_nuts sampler(model, base_rng);
        outputer.set_output_names(s, sampler, model, iter_param_names, sampler_param_names);
        outputer.init_sampler_params(sampler_params, iter_save);
        outputer.init_iter_params(iter_params, iter_save);
        if (!append_samples) {
          outputer.print_sample_names();
          outputer.output_diagnostic_names(s, sampler, model);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_max_depth(max_treedepth);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_um_nuts, Model, RNG>(sampler, num_warmup, num_iterations, num_thin, 
                                            refresh, save_warmup, 
                                            outputer,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, iter_params, adaptation_info,
                                            base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        outputer.output_adapt_finish(sampler, adaptation_info);
        // Sampling
        start = clock();
        sample_phase<a_um_nuts, Model, RNG>(sampler, num_warmup, num_iterations, num_thin, 
                                            refresh, true, 
                                            outputer,
                                            s, model, chains, iter_save_i, 
                                            qoi_idx, mean_pars, mean_lp, 
                                            sampler_params, iter_params, adaptation_info,
                                            base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } else {
        // Unit Metric HMC with Static Integration Time
        args.sampler = "HMC";
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        typedef stan::mcmc::adapt_unit_e_static_hmc<Model, RNG> a_um_hmc;
        a_um_hmc sampler(model, base_rng);
        outputer.set_output_names(s, sampler, model, iter_param_names, sampler_param_names);
        outputer.init_sampler_params(sampler_params, iter_save);
        outputer.init_iter_params(iter_params, iter_save);
        if (!append_samples) {
          outputer.print_sample_names();
          outputer.output_diagnostic_names(s, sampler, model);
        } 
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else sampler.set_nominal_stepsize(epsilon);
        sampler.set_stepsize_jitter(epsilon_pm);
        sampler.set_nominal_stepsize_and_L(epsilon, leapfrog_steps);
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        clock_t start = clock();
        warmup_phase<a_um_hmc, Model, RNG>(sampler, num_warmup, num_iterations, num_thin, 
                                           refresh, save_warmup, 
                                           outputer,
                                           s, model, chains, iter_save_i, 
                                           qoi_idx, mean_pars, mean_lp, 
                                           sampler_params, iter_params, adaptation_info,
                                           base_rng); 
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        sampler.disengage_adaptation();
        outputer.output_adapt_finish(sampler, adaptation_info);
        // Sampling
        start = clock();
        sample_phase<a_um_hmc, Model, RNG>(sampler, num_warmup, num_iterations, num_thin, 
                                           refresh, true, 
                                           outputer,
                                           s, model, chains, iter_save_i, 
                                           qoi_idx, mean_pars, mean_lp, 
                                           sampler_params, iter_params, adaptation_info,
                                           base_rng); 
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
      } 
      std::cout << std::endl;
      if (iter_save_wo_warmup > 0) {
        mean_lp /= iter_save_wo_warmup;
        for (std::vector<double>::iterator it = mean_pars.begin();
             it != mean_pars.end(); 
             ++it) 
          (*it) /= iter_save_wo_warmup;
      } 
      if (refresh > 0) { 
        outputer.print_timing(warmDeltaT, sampleDeltaT, &std::cout);
      }
      
      outputer.output_timing(warmDeltaT, sampleDeltaT);
      if (sample_file_flag) {
        std::cout << "Sample of chain " 
                         << chain_id 
                         << " is written to file " << sample_file << "."
                         << std::endl;
        sample_stream.close();
      }
      if (diagnostic_file_flag) 
        diagnostic_stream.close();
     
      holder.chains = chains;
      holder.test_grad = false;
      holder.args = args;
      holder.inits = initv;
      holder.mean_pars = mean_pars; 
      holder.mean_lp__ = mean_lp;
      holder.adaptation_info = adaptation_info;
      // put sampler parameters such as treedepth together with iter_params 
      iter_params.insert(iter_params.end(), sampler_params.begin(), sampler_params.end());
      iter_param_names.insert(iter_param_names.end(),
                              sampler_param_names.begin(),
                              sampler_param_names.end());
      holder.sampler_params = iter_params;
      holder.sampler_param_names = iter_param_names;
      holder.chain_names = fnames_oi;
      return 0;
    }

  /* stan_fit proper */
  template <class Model, class RNG> 
  class stan_fit {

  private:
    pystan::io::py_var_context data_;
    Model model_;
    RNG base_rng; 
    const std::vector<std::string> names_;
    const std::vector<std::vector<unsigned int> > dims_; 
    const unsigned int num_params_; 

    std::vector<std::string> names_oi_; // parameters of interest 
    std::vector<std::vector<unsigned int> > dims_oi_; 
    std::vector<size_t> names_oi_tidx_;  // the total indexes of names2.
    // std::vector<size_t> midx_for_col2row; // indices for mapping col-major to row-major
    std::vector<unsigned int> starts_oi_;  
    unsigned int num_params2_;  // total number of POI's.   
    std::vector<std::string> fnames_oi_; 

  private: 
    /**
     * Tell if a parameter name is an element of an array parameter. 
     * Note that it only supports full specified name; slicing 
     * is not supported. The test only tries to see if there 
     * are brackets. 
     */
    bool is_flatname(const std::string& name) {
      return name.find('[') != name.npos && name.find(']') != name.npos; 
    } 
  
    /*
     * Update the parameters we are interested for the model. 
     * As well, the dimensions vector for the parameters are 
     * updated. 
     */
    void update_param_oi0(const std::vector<std::string>& pnames) {
      names_oi_.clear(); 
      dims_oi_.clear(); 
      names_oi_tidx_.clear(); 

      std::vector<unsigned int> starts; 
      calc_starts(dims_, starts);
      for (std::vector<std::string>::const_iterator it = pnames.begin(); 
           it != pnames.end(); 
           ++it) { 
        size_t p = find_index(names_, *it); 
        if (p != names_.size()) {
          names_oi_.push_back(*it); 
          dims_oi_.push_back(dims_[p]); 
          if (*it == "lp__") {
            names_oi_tidx_.push_back(-1); // -1 for lp__ as it is not really a parameter  
            continue;
          } 
          size_t i_num = calc_num_params(dims_[p]); 
          size_t i_start = starts[p]; 
          for (size_t j = i_start; j < i_start + i_num; j++)
            names_oi_tidx_.push_back(j);
        } 
      }
      calc_starts(dims_oi_, starts_oi_);
      num_params2_ = names_oi_tidx_.size(); 
    } 

  public:
    bool update_param_oi(std::vector<std::string> pars) {
      std::vector<std::string> pnames = pars;
      if (std::find(pnames.begin(), pnames.end(), "lp__") == pnames.end()) 
        pnames.push_back("lp__"); 
      update_param_oi0(pnames); 
      get_all_flatnames(names_oi_, dims_oi_, fnames_oi_, true); 
      return true;
    } 

    stan_fit(vars_r_t& vars_r, vars_i_t& vars_i) : data_(vars_r, vars_i),
      model_(data_), // model_(data_, &rstan::io::rcout)
      base_rng(static_cast<boost::uint32_t>(std::time(0))),
      names_(get_param_names(model_)), 
      dims_(get_param_dims(model_)), 
      num_params_(calc_total_num_params(dims_)), 
      names_oi_(names_), 
      dims_oi_(dims_),
      num_params2_(num_params_)
    {
      for (size_t j = 0; j < num_params2_ - 1; j++) 
        names_oi_tidx_.push_back(j);
      names_oi_tidx_.push_back(-1); // lp__
      calc_starts(dims_oi_, starts_oi_);
      get_all_flatnames(names_oi_, dims_oi_, fnames_oi_, true); 
      // get_all_indices_col2row(dims_, midx_for_col2row);
    }             

    /**
     * Transform the parameters from its defined support
     * to unconstrained space 
     * 
     */
    std::vector<double> unconstrain_pars(vars_r_t& vars_r, vars_i_t& vars_i) {
      pystan::io::py_var_context par_context(vars_r, vars_i);
      std::vector<int> params_i;
      std::vector<double> params_r;
      model_.transform_inits(par_context, params_i, params_r);
      return params_r;
    }

    /**
     * Contrary to unconstrain_pars, transform parameters
     * from unconstrained support to the constrained. 
     *
     * @param params_r The real parameters on the unconstrained 
     *  space. 
     * 
     */ 
    std::vector<double> constrain_pars(std::vector<double>& params_r) {
      std::vector<double> par;
      if (params_r.size() != model_.num_params_r()) {
        std::stringstream msg; 
        msg << "Number of unconstrained parameters does not match " 
               "that of the model (" 
            << params_r.size() << " vs " 
            << model_.num_params_r() 
            << ").";
        throw std::domain_error(msg.str()); 
      } 
      std::vector<int> params_i(model_.num_params_i());
      model_.write_array(base_rng, params_r, params_i, par);
      return par;
    } 

    /**
     * Expose the log_prob of the model to stan_fit so R user
     * can call this function. 
     * 
     * @param upar The real parameters on the unconstrained 
     *  space. 
     */
    double log_prob(std::vector<double> upar, bool jacobian_adjust_transform, bool gradient) {
      using std::vector;
      vector<double> par_r = upar;
      if (par_r.size() != model_.num_params_r()) {
        std::stringstream msg; 
        msg << "Number of unconstrained parameters does not match " 
               "that of the model (" 
            << par_r.size() << " vs " 
            << model_.num_params_r() 
            << ").";
        throw std::domain_error(msg.str()); 
      } 
      vector<int> par_i(model_.num_params_i(), 0);
      if (!gradient) { 
        if (jacobian_adjust_transform) {
          return stan::model::log_prob_propto<true>(model_, par_r, par_i, &std::cout);
        } else {
          return stan::model::log_prob_propto<false>(model_, par_r, par_i, &std::cout);
        } 
      } 

      std::vector<double> grad; 
      double lp;
      if (jacobian_adjust_transform)
        lp = stan::model::log_prob_grad<true,true>(model_, par_r, par_i, grad, &std::cout);
      else 
        lp = stan::model::log_prob_grad<true,false>(model_, par_r, par_i, grad, &std::cout);
      // RStan returns the gradient as an attribute. Python numbers don't have attributes.
      return lp;
    } 

    /**
     * Expose the grad_log_prob of the model to stan_fit so R user
     * can call this function. 
     * 
     * @param upar The real parameters on the unconstrained 
     *  space. 
     * @param jacobian_adjust_transform TRUE/FALSE, whether
     *  we add the term due to the transform from constrained
     *  space to unconstrained space implicitly done in Stan.
     */
    std::vector<double> grad_log_prob(std::vector<double> upar, bool jacobian_adjust_transform) {
      std::vector<double> par_r = upar;
      if (par_r.size() != model_.num_params_r()) {
        std::stringstream msg; 
        msg << "Number of unconstrained parameters does not match " 
               "that of the model (" 
            << par_r.size() << " vs " 
            << model_.num_params_r() 
            << ").";
        throw std::domain_error(msg.str()); 
      } 
      std::vector<int> par_i(model_.num_params_i(), 0);
      std::vector<double> gradient; 
      double lp;
      if (jacobian_adjust_transform)
        lp = stan::model::log_prob_grad<true,true>(model_, par_r, par_i, gradient, &std::cout);
      else 
        lp = stan::model::log_prob_grad<true,false>(model_, par_r, par_i, gradient, &std::cout);
      // RStan returns the lp as an attribute. Python numbers don't have attributes.
      return gradient;
    } 

    /**
     * Return the number of unconstrained parameters 
     */ 
    int num_pars_unconstrained() {
      int n = model_.num_params_r();
      return n;
    } 
    
    int call_sampler(PyStanArgs& args, PyStanHolder& holder) { 
      int ret;
      ret = sampler_command(args, model_, holder, names_oi_tidx_, 
                            fnames_oi_, base_rng);
      if (ret != 0) {
        throw std::runtime_error("Something went wrong after call_sampler.");
      } 
      return ret; // FIXME: rstan returns holder
    } 

    std::vector<std::string> param_names() const {
       return names_;
    } 

    std::vector<std::string> param_names_oi() const {
       return names_oi_;
    } 

    std::vector<std::vector<unsigned int> > param_dims() const {
        return dims_;
    } 

    std::vector<std::string> param_fnames_oi() const {
       std::vector<std::string> fnames; 
       get_all_flatnames(names_oi_, dims_oi_, fnames, true); 
       return fnames; 
    } 

  };
} 
#endif 
