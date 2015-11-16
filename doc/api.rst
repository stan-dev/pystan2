.. _api:

.. currentmodule:: pystan

API
===

.. autosummary::
   stan
   stanc
   StanModel

:ref:`StanFit4model <StanFit4model>` instances are also documented on this page.


.. automodule:: pystan
   :members: stan, stanc, StanModel

.. _StanFit4model:

StanFit4model
-------------

Each StanFit instance is model-specific, so the name of the class will be
something like: ``StanFit4anon_model``. The  ``StanFit4model`` instances expose
a number of methods.

.. NB: Documentation here needs to be kept in sync with code.

.. py:class:: StanFit4model

   .. py:method:: plot(pars=None)

      Visualize samples from posterior distributions

      Parameters

      pars : sequence of str
          names of parameters

      This is currently an alias for the `traceplot` method.

   .. py:method:: extract(pars=None, permuted=True, inc_warmup=False)

      Extract samples in different forms for different parameters.

      Parameters

      pars : sequence of str
          names of parameters (including other quantities)
      permuted : bool
          If True, returned samples are permuted. All chains are merged and
          warmup samples are discarded.
      inc_warmup : bool
         If True, warmup samples are kept; otherwise they are discarded. If
         `permuted` is True, `inc_warmup` is ignored.

      Returns

      samples : dict or array
      If `permuted` is True, return dictionary with samples for each
      parameter (or other quantity) named in `pars`.

      If `permuted` is False, an array is returned. The first dimension of
      the array is for the iterations; the second for the number of chains;
      the third for the parameters. Vectors and arrays are expanded to one
      parameter (a scalar) per cell, with names indicating the third dimension.
      Parameters are listed in the same order as `model_pars` and `flatnames`.

   .. py:method:: log_prob(upar, adjust_transform=True, gradient=False)

      Expose the log_prob of the model to stan_fit so user can call
      this function.

      Parameters

      upar :
          The real parameters on the unconstrained space.
      adjust_transform : bool
          Whether we add the term due to the transform from constrained
          space to unconstrained space implicitly done in Stan.

      Note

      In Stan, the parameters need be defined with their supports. For
      example, for a variance parameter, we must define it on the positive
      real line. But inside Stan's sampler, all parameters defined on the
      constrained space are transformed to unconstrained space, so the log
      density function need be adjusted (i.e., adding the log of the absolute
      value of the Jacobian determinant).  With the transformation, Stan's
      samplers work on the unconstrained space and once a new iteration is
      drawn, Stan transforms the parameters back to their supports. All the
      transformation are done inside Stan without interference from the users.
      However, when using the log density function for a model exposed to
      Python, we need to be careful.  For example, if we are interested in
      finding the mode of parameters on the constrained space, we then do not
      need the adjustment.  For this reason, there is an argument named
      `adjust_transform` for functions `log_prob` and `grad_log_prob`.

   .. py:method:: grad_log_prob(upars, adjust_transform=True)

      Expose the grad_log_prob of the model to stan_fit so user
      can call this function.

      Parameters

      upar : array
          The real parameters on the unconstrained space.
      adjust_transform : bool
          Whether we add the term due to the transform from constrained
          space to unconstrained space implicitly done in Stan.


   .. py:method:: get_adaptation_info()

      Obtain adaptation information for sampler, which now only NUTS2 has.

      The results are returned as a list, each element of which is a character
      string for a chain.

   .. py:method:: get_logposterior(inc_warmup=True)

      Get the log-posterior (up to an additive constant) for all chains.

      Each element of the returned array is the log-posterior for
      a chain. Optional parameter `inc_warmup` indicates whether to
      include the warmup period.


   .. py:method:: get_sampler_params(inc_warmup=True)

      Obtain the parameters used for the sampler such as `stepsize` and
      `treedepth`. The results are returned as a list, each element of which
      is an OrderedDict a chain. The dictionary has number of elements
      corresponding to the number of parameters used in the sampler. Optional
      parameter `inc_warmup` indicates whether to include the warmup period.

   .. py:method:: get_posterior_mean()

      Get the posterior mean for all parameters

      Returns

      means : array of shape (num_parameters, num_chains)
          Order of parameters is given by self.model_pars or self.flatnames
          if parameters of interest include non-scalar parameters. An additional
          column for mean lp__ is also included.

   .. py:method:: unconstrain_pars(par)

      Transform parameters from defined support to unconstrained space

   .. py:method:: get_seed()

   .. py:method:: get_inits()

   .. py:method:: get_stancode()
