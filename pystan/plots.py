import logging

logger = logging.getLogger('pystan')


def traceplot(fit, pars, dtypes, **kwargs):
    """
    Use pymc's traceplot to display parameters.

    Additional arguments are passed to pymc.plots.traceplot.
    """
    # FIXME: eventually put this in the StanFit object
    # FIXME: write a to_pymc(_trace) function
    logger.warning("Deprecation warning, plotting module"\
                   " is going to be removed from PyStan (2.19<=)."\
                   " In future, use ArviZ library (`pip install arviz`)")
    try:
        from pystan.external.pymc import plots
    except ImportError:
        logger.critical("matplotlib required for plotting.")
        raise
    if pars is None:
        pars = list(fit.model_pars) + ["lp__"]
    return plots.traceplot(fit.extract(dtypes=dtypes, pars=pars, permuted=False), pars, **kwargs)
