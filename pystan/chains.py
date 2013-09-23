import pystan._chains as _chains


def effective_sample_size(sim, n):
    """Calculate effective sample size

    Parameters
    ----------
    sim : chains
    n : int
        Chain index starting from 0
    """
    return _chains.effective_sample_size(sim, n)


def split_potential_scale_reduction(sim, n):
    """Calculate rhat

    Parameters
    ----------
    sim : chains
    n : int
        Chain index starting from 0
    """
    return _chains.split_potential_scale_reduction(sim, n)
