.. _mass_matrix:

.. currentmodule:: pystan

============
 Mass matrix
============

Mass matrix is one part of the tuned parameters for hmc. See
`hmc-algorithm-parameters <https://mc-stan.org/docs/2_19/reference-manual/hmc-algorithm-parameters.html>`_.

Two examples are shown where it is possible to use pre-tuned mass-matrix with other
pre-tuned parameters.

Example 1: (pseudo-)continue sampling
-------------------------------------
The first example shows an example how to (pseudo-)continue chains to get more draws to old fit.
This procedure needs some manual processing and ArviZ is recommended for further analysis.
To make this reproducible, user can manually set seed for each fit.

.. code-block:: python

    from pystan import StanModel

    # bernoulli model
    model_code = """
        data {
          int<lower=0> N;
          int<lower=0,upper=1> y[N];
        }
        parameters {
          real<lower=0,upper=1> theta;
        }
        model {
          theta ~ beta(0.5, 0.5);  // Jeffreys' prior
          for (n in 1:N)
            y[n] ~ bernoulli(theta);
        }
    """

    data = dict(N=10, y=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1])
    sm = StanModel(model_code=model_code)
    fit = sm.sampling(data=data)

    # reuse tuned parameters
    stepsize = fit.get_stepsize()
    mass_matrix = fit.get_mass_matrix()
    init = fit.get_last_position()

    mass_matrix_dict = dict(enumerate(mass_matrix))

    control = {"stepsize" : stepsize, "mass_matrix" : mass_matrix_dict}
    fit2 = sm.sampling(data=data, warmup=0, iter=1000, control=control, init=init)

User needs to combine the draws manually. This can be done with ``numpy.concatenate``
and the resulting combined draws can be further analyzed with ArviZ.

.. code-block:: python

    # combine arrays with numpy
    extract1 = fit1.extract()
    extract2 = fit2.extract()

    combined = {}
    for key in extract1.keys():
        arr1 = extract1[key]
        arr2 = extract2[key]
        combined[key] = np.concatenate((arr1, arr2), axis=0)

    import arviz as az
    # transform shape order from (draw, chain, *shape) to (chain, draw, *shape)
    for key in combined.items():
        combined[key] = np.swapaxes(combined[key], 0, 1)
    inference_data = az.from_dict(posterior=combined)

    # summary
    print(az.summary(inference_data, index_origin=1))

    # traceplot
    az.plot_trace(inference_data)

Example 2: Pretune hmc parameters
---------------------------------
The seconds example goes through example how to use pre-tuned variables. With large and slow models,
it is sometimes convenient to reuse the prelearned tuning parameters.
This enables user to skip warmup period for new chains. Example code will first run warmup for
one chain and then reuse the mass matrix, stepsize and last sample location for further chains.

.. code-block:: python

    from pystan import StanModel

    # bernoulli model
    model_code = """
        data {
          int<lower=0> N;
          int<lower=0,upper=1> y[N];
        }
        parameters {
          real<lower=0,upper=1> theta;
        }
        model {
          theta ~ beta(0.5, 0.5);  // Jeffreys' prior
          for (n in 1:N)
            y[n] ~ bernoulli(theta);
        }
    """

    data = dict(N=10, y=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1])
    sm = StanModel(model_code=model_code)
    fit_warmup = sm.sampling(data=data, chains=1, warmup=500, iter=501, check_hmc_diagnostics=False)

    # reuse tuned parameters
    stepsize = fit_warmup.get_stepsize()[0] # select chain 1
    mass_matrix = fit_warmup.get_mass_matrix()[0] # select chain 1
    last_position = fit_warmup.get_last_position()[0] # select chain 1
    chains = 4
    init = [last_position for _ in range(chains)]

    control = {"stepsize" : stepsize, "mass_matrix" : mass_matrix}
    fit = sm.sampling(data=data, warmup=0, chains=4, iter=1000, control=control, init=init)
    print(fit)


Shape and usage
---------------
The shape of the inserted mass matrix changes depending on the used algorithm.
The default algorithm with NUTS is ``diag_e``, which means that only the diagonal
of the mass matrix is used. This also means that the shape of the mass matrix is
``(n_flatnames,)``. With other algorithm ``dense_e`` the full matrix is defined
and the mass matrix shape is ``(n_flatnames, n_flatnames)``. The mass matrix needs
to be strictly positive definite.

The mass matrix is given for the ``.sampling`` method inside the ``control`` dictionary.
The ``mass_matrix`` can be either iterable (list, tuple, ndarray), dictionary of iterable
or string (path to Rdump or JSON file with parameter "inv_metric" and values from the iterable).
