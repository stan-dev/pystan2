#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

import io
import logging
import random

import pystan._api  # stanc wrapper
from pystan._compat import string_types
from pystan.constants import MAX_UINT
from pystan.model import StanModel

logger = logging.getLogger('pystan')
logger.setLevel(logging.INFO)


def stanc(file=None, charset='utf-8', model_code=None, model_name="anon_model",
          verbose=False, obsfucate_model_name=False):
    """Translate Stan model specification into C++ code.

    Parameters
    ----------
    file : {string, file}, optional
        If filename, the string passed as an argument is expected to
        be a filename containing the Stan model specification.

        If file, the object passed must have a 'read' method (file-like
        object) that is called to fetch the Stan model specification.

    charset : string, 'utf-8' by default
        If bytes or files are provided, this charset is used to decode.

    model_code : string, optional
        A string containing the Stan model specification. Alternatively,
        the model may be provided with the parameter `file`.

    model_name: string, 'anon_model' by default
        A string naming the model. If none is provided 'anon_model' is
        the default. However, if `file` is a filename, then the filename
        will be used to provide a name.

    verbose : boolean, False by default
        Indicates whether intermediate output should be piped to the
        console. This output may be useful for debugging.

    obsfucate_model_name : boolean, True by default
        If False the model name in the generated C++ code will not be made
        unique by the insertion of randomly generated characters.
        Generally it is recommended that this parameter be left as True.

    Returns
    -------
    stanc_ret : dict
        A dictionary with the following keys: model_name, model_code,
        cpp_code, and status. Status indicates the success of the translation
        from Stan code into C++ code (success = 0, error = -1).

    Notes
    -----
    C++ reserved words and Stan reserved words may not be used for
    variable names; see the Stan User's Guide for a complete list.

    See also
    --------
    StanModel : Class representing a compiled Stan model
    stan : Fit a model using Stan

    References
    ----------
    The Stan Development Team (2013) *Stan Modeling Language User's
    Guide and Reference Manual*.  <http://mc-stan.org/>.

    Examples
    --------
    >>> stanmodelcode = '''
    ... data {
    ...   int<lower=0> N;
    ...   real y[N];
    ... }
    ...
    ... parameters {
    ...   real mu;
    ... }
    ...
    ... model {
    ...   mu ~ normal(0, 10);
    ...   y ~ normal(mu, 1);
    ... }
    ... '''
    >>> r = stanc(model_code=stanmodelcode, model_name = "normal1")
    >>> sorted(r.keys())
    ['cppcode', 'model_code', 'model_cppname', 'model_name', 'status']
    >>> r['model_name']
    'normal1'

    """
    if file and model_code:
        raise ValueError("Specify stan model with `file` or `model_code`, "
                         "not both.")
    if file is None and model_code is None:
        raise ValueError("Model file missing and empty model_code.")
    if file is not None:
        if isinstance(file, string_types):
            try:
                with io.open(file, 'rt', encoding='utf-8') as f:
                    model_code = f.read()
            except:
                logger.critical("Unable to read file specified by `file`.")
                raise
        else:
            model_code = file.read()

    # bytes, going into C++ code
    model_name_bytes = model_name.encode('ascii')
    model_code_bytes = model_code.encode('ascii')

    result = pystan._api.stanc(model_code_bytes, model_name_bytes)
    if result['status'] == -1:  # EXCEPTION_RC = -1
        logger.error("Error parsing:\n{}".format(result['msg']))
    del result['msg']
    result.update({'model_name': model_name})
    result.update({'model_code': model_code})
    return result


def stan(file=None, model_name="anon_model", model_code=None, fit=None,
         data=None, pars=None, chains=4, iter=2000, warmup=None, thin=1,
         init="random", seed=random.randint(0, MAX_UINT), sample_file=None,
         diagnostic_file=None, save_dso=True, verbose=False, boost_lib=None,
         eigen_lib=None, **kwargs):
    """Fit a model using Stan.

    Parameters
    ----------

    file : string {'filename', file-like object}
        Model code must found via one of the following parameters: `file` or
        `model_code`.

        If `file` is a filename, the string passed as an argument is expected
        to be a filename containing the Stan model specification.

        If `file` is a file object, the object passed must have a 'read' method
        (file-like object) that is called to fetch the Stan model specification.

    charset : string, optional
        If bytes or files are provided, this charset is used to decode. 'utf-8'
        by default.

    model_code : string
        A string containing the Stan model specification. Alternatively,
        the model may be provided with the parameter `file`.

    model_name: string, optional
        A string naming the model. If none is provided 'anon_model' is
        the default. However, if `file` is a filename, then the filename
        will be used to provide a name. 'anon_model' by default.

    fit : StanFit instance
        An instance of StanFit derived from a previous fit, None by
        default. If `fit` is not None, the compiled model associated
        with a previous fit is reused and recompilation is avoided.

    data : dict
        A Python dictionary providing the data for the model. Variables
        for Stan are stored in the dictionary as expected. Variable
        names are the keys and the values are their associated values.
        Stan only accepts certain kinds of values; see Notes.

    pars : list of string, optional
        A list of strings indicating parameters of interest. By default
        all parameters specified in the model will be stored.

    chains : int, optional
        Positive integer specifying number of chains. 4 by default.

    iter : int, 2000 by default
        Positive integer specifying how many iterations for each chain
        including warmup.

    warmup : int, iter//2 by default
        Positive integer specifying number of warmup (aka burin) iterations.
        As `warmup` also specifies the number of iterations used for step-size
        adaption, warmup samples should not be used for inference.

    thin : int, optional
        Positive integer specifying the period for saving samples.
        Default is 1.

    init : {0, '0', 'random', function returning dict, list of dict}, optional
        Specifies how initial parameter values are chosen: 0 or '0'
        initializes all to be zero on the unconstrained support; 'random'
        generates random initial values; list of size equal to the number
        of chains (`chains`), where the list contains a dict with initial
        parameter values; function returning a dict with initial parameter
        values. The function may take an optional argument `chain_id`.

    seed : int, optional
        The seed, a positive integer for random number generation. Only
        one seed is needed when multiple chains are used, as the other
        chain's seeds are generated from the first chain's to prevent
        dependency among random number streams. By default, seed is
        ``random.randint(0, MAX_UINT)``.

    sample_file : string, optional
        File name specifying where samples for *all* parameters and other
        saved quantities will be written. If not provided, no samples
        will be written. If the folder given is not writable, a temporary
        directory will be used. When there are multiple chains, an underscore
        and chain number are appended to the file name. By default do not
        write samples to file.

    diagnostic_file : string, optional
        File name specifying where diagnostic information should be written.
        By default no diagnostic information is recorded.

    boost_lib : string, optional
        The path to a version of the Boost C++ library to use instead of
        the one supplied with PyStan.

    eigen_lib : string, optional
        The path to a version of the Eigen C++ library to use instead of
        the one in the supplied with PyStan.

    save_dso : boolean, optional
        Indicates whether the dynamic shared object (DSO) compiled from
        C++ code will be saved for use in a future Python session. True by
        default.

    verbose : boolean, optional
        Indicates whether intermediate output should be piped to the console.
        This output may be useful for debugging. False by default.


    """
    # NOTE: this is a thin wrapper for other functions. Error handling occurs
    # elsewhere.
    if data is None:
        data = {}
    if warmup is None:
        warmup = int(iter // 2)
    if fit is not None:
        m = fit.stanmodel
    else:
        m = StanModel(file=file, model_name=model_name, model_code=model_code,
                      boost_lib=boost_lib, eigen_lib=eigen_lib,
                      save_dso=save_dso, verbose=verbose, **kwargs)
    if sample_file is not None:
        raise NotImplementedError
    fit = m.sampling(data, pars, chains, iter, warmup, thin, seed, init,
                     sample_file=sample_file, diagnostic_file=diagnostic_file,
                     verbose=verbose, **kwargs)
    return fit
