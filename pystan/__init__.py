#-----------------------------------------------------------------------------
# Copyright (c) 2013-2014 PyStan developers
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------
import logging

from pystan.api import stanc, stan
from pystan.misc import read_rdump, stan_rdump
from pystan.model import StanModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pystan')
logger.addHandler(logging.NullHandler())

# following PEP 386
__version__ = "2.5.0.0"
