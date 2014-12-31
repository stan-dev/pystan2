MAX_UINT = 2**31 - 1  # simpler than ctypes.uint(-1).value, I think
EPSILON = 1e-6

try:
    from enum import Enum  # Python 3.4
except ImportError:
    from pystan.external.enum import Enum

sampling_algo_t = Enum('sampling_algo_t', 'NUTS HMC Metropolis Fixed_param')


class optim_algo_t(Enum):
    Newton = 1
    BFGS = 3
    LBFGS = 4

sampling_metric_t = Enum('sampling_metric_t', 'UNIT_E DIAG_E DENSE_E')
stan_args_method_t = Enum('stan_args_method_t', 'SAMPLING OPTIM TEST_GRADIENT')
