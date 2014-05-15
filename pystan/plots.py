import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pystan')


def traceplot(fit, pars):
    """Use pymc's traceplot to display parameters"""
    # FIXME: eventually put this in the StanFit object
    # FIXME: write a to_pymc(_trace) function
    try:
        from pystan.external.pymc import plots
    except ImportError:
        logger.critical("matplotlib required for plotting.")
        raise
    return plots.traceplot(fit.extract(), pars)
