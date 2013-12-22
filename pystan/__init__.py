#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------
import logging

from pystan.api import stanc, stan
from pystan.model import StanModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pystan')
logger.addHandler(logging.NullHandler())

# make sure this matches version found in setup.py
__version__ = "2.1.0.0"
