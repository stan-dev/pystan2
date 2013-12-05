import pystan._chains as _chains


def ess(sim, n):
    """Calculate effective sample size

    Parameters
    ----------
    sim : chains
    n : int
        Chain index starting from 0
    """
    try:
      return _chains.effective_sample_size(sim, n)
    except:
      raise Exception("Effective sample size calculation failed" \
		      + "\n- was the parameter index (n) in range?")


def splitrhat(sim, n):
    """Calculate rhat

    Parameters
    ----------
    sim : chains
    n : int
        Chain index starting from 0
    """
    return _chains.split_potential_scale_reduction(sim, n)
