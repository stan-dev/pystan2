import logging


def traceplot(fit, vars=None):
    """Use pymc's traceplot to display parameters"""
    # FIXME: eventually put this in the StanFit object
    # FIXME: write a to_pymc(_trace) function
    samples = fit.extract()
    if vars is None:
        vars = [v for v in samples.keys() if v != 'lp__']
    try:
        from pystan.external import plots
    except ImportError:
        logging.critical("matplotlib required for plotting.")
        raise
    return plots.traceplot(fit.extract(), vars)
