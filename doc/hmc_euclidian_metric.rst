.. _hmc_euclidian_metric:

.. currentmodule:: pystan

======================
 HMC: Euclidian Metric
======================

Euclidian Metric (also known as mass matrix) is one of the tuned parameters for hmc algorithm. See
`hmc-algorithm-parameters <https://mc-stan.org/docs/2_19/reference-manual/hmc-algorithm-parameters.html>`_.

Two examples are shown where it is possible to use pre-tuned metric-matrix with other
pre-tuned parameters.

-------------------------------------
Example 1: (pseudo-)continue sampling
-------------------------------------

The first example shows an example how to (pseudo-)continue chains to get more draws to old fit.
The method is described as pseudo due to fact that continued sampling does not equal sampling
done in one step. With user provided seed, this procedure is reproducible.
This procedure needs some manual processing and ArviZ is recommended for further analysis.
To make this reproducible, user can manually set seed for each fit.

.. code-block:: python

    from pystan import StanModel
    from pystan.constants import MAX_UINT

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
    # initial seed can also be chosen by user
    # MAX_UINT = 2147483647
    seed = np.random.randint(0, MAX_UINT, size=1)
    fit = sm.sampling(data=data, seed=seed)

    # reuse tuned parameters
    stepsize = fit.get_stepsize()
    inv_metric = fit.get_inv_metric()
    init = fit.get_last_position()

    # use chain order as a key
    inv_metric_dict = dict(enumerate(inv_metric))

    # increment seed by 1
    seed2 = seed + 1

    control = {"stepsize" : stepsize, "inv_metric" : inv_metric_dict}
    fit2 = sm.sampling(data=data,
                       warmup=0,
                       iter=1000,
                       control=control,
                       init=init,
                       seed=seed2)

User needs to combine the draws manually. This can be done with ``numpy.concatenate``
or with ``arviz.concat`` with ``dim="draw"`` option. Combined draws can be further analyzed with ArviZ.

Case 1: ``numpy.concatenate``
=============================

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
    summary = az.summary(inference_data, index_origin=1)
    print(summary) # pandas.DataFrame

    # traceplot
    az.plot_trace(inference_data)


Case 2: ``arviz.concat``
========================

.. code-block:: python

    # Needs ArviZ version 0.4.2+
    import arviz as az
    # create InferenceData objects
    inference_data1 = az.from_pystan(fit1)
    inference_data2 = az.from_pystan(fit2)

    inference_data = az.concat(inference_data1, inference_data2, dim="draw")

    # summary
    summary = az.summary(inference_data, index_origin=1)
    print(summary) # pandas.DataFrame'

    # traceplot
    az.plot_trace(inference_data)


---------------------------------
Example 2: Pretune hmc parameters
---------------------------------
The seconds example goes through the process how to use pre-tuned variables. With large and slow models,
it is sometimes convenient to reuse the prelearned tuning parameters.
This enables user to skip warmup period for new chains. Example code will first run warmup for
one chain and then reuse the inverse metric matrix, stepsize and last sample location for further chains.

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
    inv_metric = fit_warmup.get_inv_metric()[0] # select chain 1
    last_position = fit_warmup.get_last_position()[0] # select chain 1
    chains = 4
    init = [last_position for _ in range(chains)]

    control = {"stepsize" : stepsize, "inv_metric" : inv_metric}
    fit = sm.sampling(data=data, warmup=0, chains=4, iter=1000, control=control, init=init)
    print(fit)


Shape and usage
---------------
The shape of the inserted inverse metric changes depending on the used algorithm.
The default algorithm with NUTS is ``diag_e``, which means that only the diagonal
of the inverse metric matrix is used. This also means that the shape of the inverse metric is
``(n_flatnames,)``. With other algorithm ``dense_e`` the full matrix is defined
and the inverse metric matrix shape is ``(n_flatnames, n_flatnames)``. The inverse
metric matrix needs to be strictly positive definite.

The inverse metric matrix is given for the ``.sampling`` method inside the ``control`` dictionary.
The ``inv_metric`` can be either iterable (list, tuple, ndarray), dictionary of iterable with chain
order as the key or ``inv_metric`` can be a string (path to Rdump or JSON file with a
parameter "inv_metric").
