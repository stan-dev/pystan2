#-----------------------------------------------------------------------------
# Copyright (c) 2013-2015 PyStan developers
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------
import logging
import platform

from pystan.api import stanc, stan
from pystan.misc import read_rdump, stan_rdump, stansummary, add_libtbb_path
from pystan.diagnostics import check_hmc_diagnostics
from pystan.model import StanModel
from pystan.lookup import lookup

logger = logging.getLogger('pystan')
logger.addHandler(logging.NullHandler())
if len(logger.handlers) == 1:
    logging.basicConfig(level=logging.INFO)

if platform.system() == "Windows":
    add_libtbb_path()

# clean imports
del add_libtbb_path
del platform

# following PEP 386
# See also https://docs.openstack.org/pbr/latest/user/semver.html
__version__ = '2.22.1.0'
