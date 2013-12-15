import logging

logger = logging.getLogger('pystan')
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


def traceplot(fit, vars=None):
    """Use pymc's traceplot to display parameters"""
    # FIXME: eventually put this in the StanFit object
    # FIXME: write a to_pymc(_trace) function
    samples = fit.extract()
    if vars is None:
        vars = [v for v in samples.keys() if v != 'lp__']
    try:
        from pystan.external.pymc import plots
    except ImportError:
        logger.critical("matplotlib required for plotting.")
        raise
    return plots.traceplot(fit.extract(), vars)
