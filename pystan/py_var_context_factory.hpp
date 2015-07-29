#ifndef PYSTAN__IO__PY_VAR_CONTEXT_FACTORY_HPP
#define PYSTAN__IO__PY_VAR_CONTEXT_FACTORY_HPP

#include <stan/interface_callbacks/var_context_factory/dump_factory.hpp>
#include <stan/interface_callbacks/var_context_factory/var_context_factory.hpp>
#include "py_var_context.hpp"

typedef std::map<std::string, std::pair<std::vector<double>, std::vector<size_t> > > vars_r_t;
typedef std::map<std::string, std::pair<std::vector<int>, std::vector<size_t> > > vars_i_t;

namespace pystan {
  namespace io {
    class py_var_context_factory
      : public stan::interface_callbacks::var_context_factory::var_context_factory<py_var_context> {
    public:
      py_var_context_factory(vars_r_t vars_r, vars_i_t vars_i) : vars_r_(vars_r), vars_i_(vars_i) { }
      py_var_context operator()(const std::string source) {
          return pystan::io::py_var_context(vars_r_, vars_i_);
      }
      vars_r_t vars_r_;
      vars_i_t vars_i_;
    };
  }
}

#endif
