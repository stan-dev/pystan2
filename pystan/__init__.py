#-----------------------------------------------------------------------------
# Copyright (c) 2013-2015 PyStan developers
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------
import logging

from pystan.api import stanc, stan
from pystan.misc import read_rdump, stan_rdump, stansummary
from pystan.diagnostics import check_mcmc_diagnostics
from pystan.model import StanModel
from pystan.lookup import lookup
from pystan.plot import traceplot as plot_traceplot,\
                        forestplot as plot_forestplot,\
                        mcmc_parcoord as plot_mcmc_parcoord,\
                        plot_fit

logger = logging.getLogger('pystan')
logger.addHandler(logging.NullHandler())
if len(logger.handlers) == 1:
    logging.basicConfig(level=logging.INFO)

# following PEP 386
__version__ = '2.17.1.0'
