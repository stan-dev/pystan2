import pystan
import pickle
import numpy as np
import logging

logger = logging.getLogger('pystan')

# Diagnostics modified from Betancourt's stan_utility.py module

def check_div(fit, verbose = True, per_chain = False):
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
    per_chain : bool, optional
        Print the number of divergent transitions in each chain

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
        divergent = np.column_stack([y['divergent__'].astype(bool) for y in sampler_params])
    except:
        raise ValueError('Cannot access divergence information from fit object')

    n_for_chains = divergent.sum(axis=0)

    n = n_for_chains.sum()

    if n > 0:

        if verbosity > 0:

            N = divergent.size
            logger.warning('{} of {} iterations ended '.format(n, N) +
                           'with a divergence ({}%).'.format(100 * n / N))

            if per_chain:
                chain_len, num_chains = divergent.shape

                for chain_num in range(num_chains):
                    if n_for_chains[chain_num] > 0:
                        logger.warning('Chain {}: {} of {} iterations ended '.format(chain_num + 1,
                                                                                     n_for_chains[chain_num],
                                                                                     chain_len) +
                                       'with a divergence ({}%).'.format(100 * n_for_chains[chain_num] /
                                                                         chain_len))

            try:
                adapt_delta = fit.stan_args[0]['ctrl']['sampling']['adapt_delta']
            except:
                logger.warning('Cannot obtain value of adapt_delta from fit object')
                adapt_delta = None

            if adapt_delta != None:
                logger.warning('Try running with adapt_delta larger than {}'.format(adapt_delta) +
                               ' to remove the divergences.')
            else:
                logger.warning('Try running with larger adapt_delta to remove the divergences.')

        return False
    else:
        if verbosity > 1:
            logger.info('No divergent transitions found.')

        return True


def check_treedepth(fit, verbose = True, per_chain = False):
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
    per_chain : bool, optional
        Print the number of prematurely ending transitions in each chain

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
        depths = np.column_stack([y['treedepth__'].astype(int) for y in sampler_params])
    except:
        raise ValueError('Cannot access tree depth information from fit object')

    try:
        max_treedepth = int(fit.stan_args[0]['ctrl']['sampling']['max_treedepth'])
    except:
        raise ValueError('Cannot obtain value of max_treedepth from fit object')

    n_for_chain =  (depths >= max_treedepth).sum(axis=0)

    n = n_for_chain.sum()

    if n > 0:
        if verbosity > 0:
            N = depths.size
            logger.warning(('{} of {} iterations saturated the maximum tree depth of {}'
                                + ' ({}%)').format(n, N, max_treedepth, 100 * n / N))

            if per_chain:
                chain_len, num_chains = depths.shape

                for chain_num in range(num_chains):
                    if n_for_chains[chain_num] > 0:
                        logger.warning('Chain {}: {} of {} saturated '.format(chain_num + 1,
                                                                              n_for_chains[chain_num],
                                                                              chain_len) +
                                       'the maximum tree depth of {} ({}%).'.format(max_treedepth,
                                                                                    100 * n_for_chains[chain_num] /
                                                                                    chain_len))

            logger.warning('Run again with max_treedepth larger than {}'.format(max_treedepth) +
                           ' to avoid saturation')

        return False
    else:
        if verbosity > 1:
            logger.info('No transitions that ended prematurely due to maximum tree depth limit')

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

    try:
        energies = np.column_stack([y['energy__'] for y in sampler_params])
    except:
        raise ValueError('energy__ not in sampler params of fit object')

    chain_len, num_chains = energies.shape

    numer = ((np.diff(energies, axis=0)**2).sum(axis=0)) / chain_len

    denom = np.var(energies, axis=0)

    e_bfmi = numer / denom

    no_warning = True
    for chain_num in range(num_chains):

        if e_bfmi[chain_num] < 0.2:
            if verbosity > 0:
                logger.warning('Chain {}: E-BFMI = {}'.format(chain_num + 1,
                                                              e_bfmi[chain_num]))

            no_warning = False
        else:
            if verbosity > 1:
                logger.info('Chain {}: E-BFMI (= {}) '.format(chain_num + 1,
                                                              e_bfmi[chain_num]) +
                            'equals or exceeds threshold of 0.2.')

    if no_warning:
        if verbosity > 1:
            logger.info('E-BFMI indicated no pathological behavior')

        return True
    else:
        if verbosity > 0:
            logger.warning('E-BFMI below 0.2 indicates you may need to reparameterize your model')

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

    n_effs = fit_summary['summary'][:, n_effs_index]
    names = fit_summary['summary_rownames']
    n_iter = sum(fit.sim['n_save'])-sum(fit.sim['warmup2'])

    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if ((ratio < 0.001) or np.isnan(ratio) or np.isinf(ratio)):
            if verbosity > 0:
                logger.warning('n_eff / iter for parameter {} is {}!'.format(name, ratio))

            no_warning = False

    if no_warning:
        if verbosity > 1:
            logger.info('n_eff / iter looks reasonable for all parameters')

        return True
    else:
        if verbosity > 0:
            logger.warning('n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated')

        return False

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

    verbosity = int(verbose)

    fit_summary = fit.summary(probs=[0.5])

    try:
        Rhat_index = fit_summary['summary_colnames'].index('Rhat')
    except:
        raise ValueError('Summary of fit object appears to lack information on Rhat')

    rhats = fit_summary['summary'][:, Rhat_index]
    names = fit_summary['summary_rownames']

    no_warning = True
    for rhat, name in zip(rhats, names):
        if (np.isnan(rhat) or np.isinf(rhat) or (rhat > 1.1) or (rhat < 0.9)):

            if verbosity > 0:
                logger.warning('Rhat for parameter {} is {}!'.format(name, rhat))

            no_warning = False

    if no_warning:
        if verbosity > 1:
            logger.info('Rhat looks reasonable for all parameters')

        return True
    else:
        if verbosity > 0:
            logger.warning('Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed')

        return False

def check_hmc_diagnostics(fit, verbose = True, per_chain = False):
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
    per_chain : bool, optional
        Where applicable, print diagnostics on a per-chain basis. This
        applies mainly to the divergence and treedepth checks.


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
            logger.warning('Skipping check of effective sample size (n_eff)')

    try:
        out_dict['Rhat'] = check_rhat(fit, verbose)
    except ValueError:
        if verbosity > 0:
            logger.warning('Skipping check of potential scale reduction factors (Rhat)')

    try:
        out_dict['divergence'] = check_div(fit, verbose, per_chain)
    except ValueError:
        if verbosity > 0:
            logger.warning('Skipping check of divergent transitions (divergence)')

    try:
        out_dict['treedepth'] = check_treedepth(fit, verbose, per_chain)
    except ValueError:
        if verbosity > 0:
            logger.warning('Skipping check of transitions ending prematurely due to maximum tree depth limit (treedepth)')

    try:
        out_dict['energy'] = check_energy(fit, verbose)
    except ValueError:
        if verbosity > 0:
            logger.warning('Skipping check of E-BFMI (energy)')

    return out_dict
