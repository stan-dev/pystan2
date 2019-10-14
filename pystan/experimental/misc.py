import datetime
import logging
import re
import os
import datetime

import pystan

logger = logging.getLogger('pystan')

def fix_include(model_code):
    """Reformat `model_code` (remove whitespace) around the #include statements.

    Note
    ----
        A modified `model_code` is returned.

    Parameters
    ----------
    model_code : str
        Model code

    Returns
    -------
    str
        Reformatted model code

    Example
    -------

    >>> from pystan.experimental import fix_include
    >>> model_code = "parameters { #include myfile.stan \n..."
    >>> model_code_reformatted = fix_include(model_code)
    # "parameters {#include myfile.stan\n..."

    """
    pattern = r"(?<=\n)\s*(#include)\s*(\S+)\s*(?=\n)"
    model_code, n = re.subn(pattern, r"\1 \2", model_code)
    if n == 1:
        msg = "Made {} substitution for the model_code"
    else:
        msg = "Made {} substitutions for the model_code"
    logger.info(msg.format(n))
    return model_code


class EvalLogProb:
    """Evaluate log-p and grad log-p without sampling step."""
    def __init__(self, data, *, model=None, fit=None, model_code=None, file=None, model_kwargs=None):
        """Wrap log_prob and grad_log_prob for Stan model with data.

        Parameters
        ----------
        data : dict
        model : StanModel, optional
            PyStan StanModel object.
        fit : StanFit4Model, optional
            PyStan fit object.
        model_code : str, optional
            model code as a string.
        file : str, optional
            path to .stan file.

        Examples:
        ---------
        > model_code = "parameters {real y;} model {y ~ normal(0,1);}"
        > data  = {}
        > func_log_prob = EvalLogProb(data, model_code=model_code)

        >>> print(func_log_prob({"y" : 0}))
        (0.0, array([0.]))

        >>> print(func_log_prob({"y" : 5}))
        (-12.5, array([-5.]))

        >>> print(func_log_prob({"y" : -5}))
        (-12.5, array([5.]))
        """
        if model is None and fit is None:
            if all(item is None for item in (model_code, file)):
                raise TypeError("Needs one defined {model, fit, model_code, file}")

        self._model = model
        self._data = data
        self._fit = fit
        self._model_code = model_code
        self._file = file

        if fit is None:
            if model is None:
                if model_kwargs is None:
                    model_kwargs = {}
                model = pystan.StanModel(file=file, model_code=model_code, **model_kwargs)
            seed = pystan.misc._check_seed(None)
            fit = model.fit_class(data, seed)

            fnames_oi = fit._get_param_fnames_oi()
            n_flatnames = len(fnames_oi)
            fit.sim = {'samples': [],
                       'chains': 0,
                       'iter': 0,
                       'warmup': 0,
                       'thin': 1,
                       'n_save': 0,
                       'warmup2': 0,
                       'permutation': [],
                       'pars_oi': fit._get_param_names_oi(),
                       'dims_oi': fit._get_param_dims_oi(),
                       'fnames_oi': fnames_oi,
                       'n_flatnames': n_flatnames}
            fit.model_name = model.model_name
            fit.model_pars = fit._get_param_names()[:-1]
            fit.par_dims = fit._get_param_dims()[:-1]
            fit.mode = 2
            fit.inits = {}
            fit.stan_args = []
            fit.stanmodel = model
            fit.date = datetime.datetime.now()

            self._model = model
            self._fit = fit
        else:
            self._model = fit.stanmodel

        self._update_doc()

    def __call__(self, pars, lp=True, grad_lp=True, adjust_transform=True):
            upars = self._fit.unconstrain_pars(pars)
            returns = []
            if lp:
                returns.append(self._fit.log_prob(upars, adjust_transform))
            if grad_lp:
                returns.append(self._fit.grad_log_prob(upars, adjust_transform))
            return tuple(returns)

    def _update_doc(self):

        par_dim_pair = ('"{}": {}'.format(item, tuple(dim) if isinstance(dim ,list) else "") for item, dim in zip(self._fit.model_pars, self._fit.par_dims))

        doc = ("Calculate log_prob and grad_log_prob for Stan model with data.\n    pars : dict\n    Model parameters (name, shape):\n        {model_pars}\n"
               "    lp : bool\n        return log_density\n    grad_lp : bool\n        return gradient of the log_density\n"
               "    adjust_transform : bool\n        Whether we add the term due to the transform from constrained\n        space to"
               " unconstrained space implicitly done in Stan.").format(model_pars="\n        ".join(par_dim_pair))

        self.__doc__ = doc

    def __repr__(self):
        return "LogProb and GradLogProb for PyStan model: {}".format(self._fit.model_name)
