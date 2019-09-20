.. _gpu_support:

.. currentmodule:: pystan

===================================
GPU Support with Pystan 2.19+
===================================

Notice! This is an experimental feature and is not tested or supported officially with PyStan 2.
Official GPU support will land with PyStan 3.

See https://github.com/stan-dev/math/wiki/OpenCL-GPU-Routines

Please follow instructions in the above url to enable OpenCL with your GPU.

Due to use of `multiprocessing` to parallelize chains, user needs to be aware of the gpu usage.
This means that each chain will use the same gpu and its cores. This can have an affect on performance.


Example
=======

.. code-block:: python

    import pystan
    import numpy as np
    import os
    import platform

    # Set CC environmental variable
    os.environ['CC'] = "g++"

    # Example model: Gaussian process
    # see https://mc-stan.org/docs/2_20/stan-users-guide/fit-gp-section.html
    stan_code = """
    data {
      int<lower=1> N;
      real x[N];
      vector[N] y;
    }
    transformed data {
      vector[N] mu = rep_vector(0, N);
    }
    parameters {
      real<lower=0> rho;
      real<lower=0> alpha;
      real<lower=0> sigma;
    }
    model {
      matrix[N, N] L_K;
      matrix[N, N] K = cov_exp_quad(x, alpha, rho);
      real sq_sigma = square(sigma);

      // diagonal elements
      for (n in 1:N)
        K[n, n] = K[n, n] + sq_sigma;

      L_K = cholesky_decompose(K);

      rho ~ inv_gamma(5, 5);
      alpha ~ std_normal();
      sigma ~ std_normal();

      y ~ multi_normal_cholesky(mu, L_K);
    }
    """

    # use clinfo to select correct IDs
    # clinfo -l
    extra_compile_args = [
        '-DSTAN_OPENCL',
        '-DOPENCL_DEVICE_ID=0',
        '-DOPENCL_PLATFORM_ID=0',
    ]

    # On Windows define GPU_LIB
    # and setup extra_link_args
    if platform.system() == "Windows":

        # add path to $CUDA or $AMDAPPSDKROOT
        GPU_LIB = "..."

        # if -lOpenCL does not work
        # user can link with (absolute) path
        # for the correct OpenCl.dll
        opencl_dll = None # "C:/Windows/System32/OpenCL.dll"

        extra_link_args = [
            '-L"{}"'.format(GPU_LIB),
            '-lOpenCL' if opencl_dll is None else opencl_dll
        ]
    else:
        extra_link_args = []

    stan_model = pystan.StanModel(
        model_code=stan_code,
        extra_compile_args=extra_compile_args
    )


    # synthetic data for the example
    np.random.seed(15)
    N = 601
    x = np.sort(np.random.rand(N))
    y = np.sin(np.linspace(-11, 11, N)) + np.random.randn(N)/10
    stan_data = {"N" : N, "x" : x, "y" : y}

    # while the model is sampled, check your GPU usage
    # with suitable activity monitor
    fit = stan_model.sampling(
        data=stan_data,
        n_jobs=1,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )

    print(fit)
