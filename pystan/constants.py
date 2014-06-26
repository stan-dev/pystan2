MAX_UINT = 2**32 - 2  # simpler than ctypes.uint(-1).value, I think

try:
    from enum import Enum  # Python 3.4
except ImportError:
    from pystan.external.enum import Enum

sampling_algo_t = Enum('sampling_algo_t', 'NUTS HMC Metropolis Fixed_param')
optim_algo_t = Enum('optim_algo_t', 'Newton Nesterov BFGS')
sampling_metric_t = Enum('sampling_metric_t', 'UNIT_E DIAG_E DENSE_E')
stan_args_method_t = Enum('stan_args_method_t', 'SAMPLING OPTIM TEST_GRADIENT')
