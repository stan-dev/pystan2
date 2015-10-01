#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# Skip tests that require large downloads over the network to save bandwith
# usage as travis workers are stateless and therefore traditional local
# disk caching does not work.
SKLEARN_SKIP_NETWORK_TESTS=1

# TODO: if Mac OS X graphics are a problem, perhaps make this setting do something
# look at SKLEARN_SKIP_NETWORK_TESTS in scikit-learn tests
export PYSTAN_SKIP_PLOT_TESTS=1

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    DISPLAY=:99.0 JOBLIB_MULTIPROCESSING=0 nosetests -w /tmp pystan
fi

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    # skip DISPLAY setting and hope for the best
    JOBLIB_MULTIPROCESSING=0 nosetests -w /tmp pystan
fi
