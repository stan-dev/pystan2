import pystan
import pickle
import numpy
import warnings

# Diagnostics modified from Betancourt's stan_utility.py module

def check_div(fit, verbose = True):
    """Check for transitions that ended with a divergence

    Parameters
    ----------
    fit : StanFit4Model object
    verbose : bool or int, optional
        If ``verbose`` is ``False`` or a nonpositive integer, no
        diagnostic messages are printed, and only the return value of
        the function conveys diagnostic information. If it is ``True``
        (the default) or an integer greater than zero, then a
        diagnostic message is printed only if there are divergent
        transitions. If it is an integer greater than 1, then extra
        diagnostic messages are printed.

    Returns
    -------
    bool
        ``True`` if there are no problems with divergent transitions
        and ``False`` otherwise.

    Raises
    ------
    ValueError
        If ``fit`` has no information about divergent transitions.

    """

    verbosity = int(verbose)
    
    sampler_params = fit.get_sampler_params(inc_warmup=False)

    try:
        divergent = [x for y in sampler_params for x in y['divergent__']]
    except:
        raise ValueError('Cannot access divergence information from fit object')
    
    n = sum(divergent)
    
    if n > 0:
        N = len(divergent)

        if verbosity > 0:
            print('{} of {} iterations ended with a divergence ({}%).'.format(n, N,
                                                                             100 * n / N))

            try:
                adapt_delta = fit.stan_args[0]['ctrl']['sampling']['adapt_delta']
            except:
                warnings.warn('Cannot obtain value of adapt_delta from fit object')
                adapt_delta = None

            if adapt_delta != None:
                print('Try running with adapt_delta larger than {}'.format(adapt_delta) +
                      ' to remove the divergences.')
            else:        
                print('Try running with larger adapt_delta to remove the divergences.')

        return False
    else:
        if verbosity > 1:
            print ('No divergent transitions found.')

        return True
        

def check_treedepth(fit, verbose = True):
    """Check for transitions that ended prematurely due to maximum tree
    depth limit

    Parameters
    ----------
    fit : StanFit4Model object
    verbose : bool or int, optional
        If ``verbose`` is ``False`` or a nonpositive integer, no
        diagnostic messages are printed, and only the return value of
        the function conveys diagnostic information. If it is ``True``
        (the default) or an integer greater than zero, then a
        diagnostic message is printed only if there are transitions
        that ended ended prematurely due to maximum tree depth
        limit. If it is an integer greater than 1, then extra
        diagnostic messages are printed.


    Returns
    -------
    bool
        ``True`` if there are no problems with tree depth and
        ``False`` otherwise.

    Raises
    ------
    ValueError
        If ``fit`` has no information about tree depth. This could
        happen if ``fit`` was generated from a sampler other than
        NUTS.

    """

    verbosity = int(verbose)
    
    sampler_params = fit.get_sampler_params(inc_warmup=False)

    try:    
        depths = [x for y in sampler_params for x in y['treedepth__']]
    except:
        raise ValueError('Cannot access tree depth information from fit object')

    try:
        max_treedepth = fit.stan_args[0]['ctrl']['sampling']['max_treedepth']
    except:
        raise ValueError('Cannot obtain value of max_treedepth from fit object')
        
    n = sum(1 for x in depths if x >= max_treedepth)
    
    if n > 0:
        if verbosity > 0:
            N = len(depths)
            print(('{} of {} iterations saturated the maximum tree depth of {}'
                   + ' ({}%)').format(n, N, max_treedepth, 100 * n / N))
            print('Run again with max_treedepth larger than {}'.format(max_treedepth) +
                  ' to avoid saturation')

        return False
    else:
        if verbosity > 1:
            print('No transitions that ended prematurely due to maximum tree depth limit')
        
        return True

def check_energy(fit, verbose = True):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)

    Parameters
    ----------
    fit : StanFit4Model object
    verbose : bool or int, optional
        If ``verbose`` is ``False`` or a nonpositive integer, no
        diagnostic messages are printed, and only the return value of
        the function conveys diagnostic information. If it is ``True``
        (the default) or an integer greater than zero, then a
        diagnostic message is printed only if there is low E-BFMI in
        one or more chains. If it is an integer greater than 1, then
        extra diagnostic messages are printed.


    Returns
    -------
    bool
        ``True`` if there are no problems with E-BFMI and ``False``
        otherwise.

    Raises
    ------
    ValueError
        If ``fit`` has no information about E-BFMI.

    """

    verbosity = int(verbose)
    
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    
    no_warning = True
    for chain_num, s in enumerate(sampler_params):

        try:
            energies = s['energy__']
        except:
            raise ValueError('energy__ not in sampler params of fit object')
            
        numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
        denom = numpy.var(energies)
        
        if numer / denom < 0.2:
            if verbosity > 0:
                print('Chain {}: E-BFMI = {}'.format(chain_num, numer / denom))
                
            no_warning = False
            
    if no_warning:
        if verbosity > 1:
            print('E-BFMI indicated no pathological behavior')
            
        return True
    else:
        if verbosity > 0:
            print('E-BFMI below 0.2 indicates you may need to reparameterize your model')
            
        return False

def check_n_eff(fit, verbose = True):
    """Checks the effective sample size per iteration

    Parameters
    ----------
    fit : StanFit4Model object
    verbose : bool or int, optional
        If ``verbose`` is ``False`` or a nonpositive integer, no
        diagnostic messages are printed, and only the return value of
        the function conveys diagnostic information. If it is ``True``
        (the default) or an integer greater than zero, then a
        diagnostic message is printed only if there are effective
        sample sizes that appear pathologically low. If it is an
        integer greater than 1, then extra diagnostic messages are
        printed.


    Returns
    -------
    bool
        ``True`` if there are no problems with effective sample size
        and ``False`` otherwise.

    Raises
    ------
    ValueError
        If the output of ``fit.summary()`` has no information about
        effective sample size (i.e., n_eff).

    """

    verbosity = int(verbose)
    
    fit_summary = fit.summary(probs=[0.5])

    try:
        n_effs_index = fit_summary['summary_colnames'].index('n_eff')
    except:
        raise ValueError('Summary of fit object appears to lack information on effective sample size')
        
    n_effs = [x[n_effs_index] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']
    n_iter = len(fit.extract()['lp__'])

    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if (ratio < 0.001):
            if verbosity > 0:
                print('n_eff / iter for parameter {} is {}!'.format(name, ratio))
                
            no_warning = False
            
    if no_warning:
        if verbosity > 1:
            print('n_eff / iter looks reasonable for all parameters')

        return True
    else:
        if verbosity > 0:
            print('n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated')

def check_rhat(fit, verbose = True):
    """Checks the potential scale reduction factors, i.e., Rhat values

    Parameters
    ----------
    fit : StanFit4Model object
    verbose : bool or int, optional
        If ``verbose`` is ``False`` or a nonpositive integer, no
        diagnostic messages are printed, and only the return value of
        the function conveys diagnostic information. If it is ``True``
        (the default) or an integer greater than zero, then a
        diagnostic message is printed only if there are Rhat values
        too far from 1. If ``verbose`` is an integer greater than 1,
        then extra diagnostic messages are printed.


    Returns
    -------
    bool
        ``True`` if there are no problems with with Rhat and ``False``
        otherwise.

    Raises
    ------
    ValueError
        If the output of ``fit.summary()`` has no information about
        Rhat.

    """
    
    from math import isnan
    from math import isinf

    verbosity = int(verbose)

    fit_summary = fit.summary(probs=[0.5])

    try:
        Rhat_index = fit_summary['summary_colnames'].index('Rhat')
    except:
        raise ValueError('Summary of fit object appears to lack information on Rhat')
    
    rhats = [x[Rhat_index] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']

    no_warning = True
    for rhat, name in zip(rhats, names):
        if (isnan(rhat) or isinf(rhat) or rhat > 1.1 or rhat < 0.9):

            if verbosity > 0:
                print('Rhat for parameter {} is {}!'.format(name, rhat))
                
            no_warning = False
            
    if no_warning:
        if verbosity > 1:
            print('Rhat looks reasonable for all parameters')

        return True
    else:
        if verbosity > 0:
            print('Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed')

        return False

def check_MCMC_diagnostics(fit, verbose = True):
    """Checks all MCMC diagnostics

    Parameters
    ----------
    fit : StanFit4Model object
    verbose : bool or int, optional
        If ``verbose`` is ``False`` or a nonpositive integer, no
        diagnostic messages are printed, and only the return value of
        the function conveys diagnostic information. If it is ``True``
        (the default) or an integer greater than zero, then diagnostic
        messages are printed only for diagnostic checks that fail. If
        ``verbose`` is an integer greater than 1, then extra
        diagnostic messages are printed.


    Returns
    -------
    out_dict : dict
        A dictionary where each key is the name of a diagnostic check,
        and the value associated with each key is a Boolean value that
        is True if the check passed and False otherwise.  Possible
        valid keys are 'n_eff', 'Rhat', 'divergence', 'treedepth', and
        'energy', though which keys are available will depend upon the
        sampling algorithm used.

    """

    # For consistency with the individual diagnostic functions
    verbosity = int(verbose)

    out_dict = {}

    try:
        out_dict['n_eff'] = check_n_eff(fit, verbose)
    except ValueError:
        if verbosity > 0:
            print('Skipping check of effective sample size (n_eff)')

    try:
        out_dict['Rhat'] = check_rhat(fit, verbose)
    except ValueError:
        if verbosity > 0:     
            print('Skipping check of potential scale reduction factors (Rhat)')

    try:
        out_dict['divergence'] = check_div(fit, verbose)
    except ValueError:        
        if verbosity > 0:     
            print('Skipping check of divergent transitions (divergence)')

    try:
        out_dict['treedepth'] = check_treedepth(fit, verbose)
    except ValueError: 
        if verbosity > 0:     
            print('Skipping check of transitions ending prematurely due to maximum tree depth limit (treedepth)')

    try:
        out_dict['energy'] = check_energy(fit, verbose)
    except ValueError:         
        if verbosity > 0:     
            print('Skipping check of E-BFMI (energy)')

    return out_dict
