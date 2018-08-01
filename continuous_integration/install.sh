#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    export DISPLAY=:99.0
    sh -e /etc/init.d/xvfb start
fi


# steps common to linux and OS X

# use Anaconda to get compiled versions of scipy and numpy,
# modified from https://gist.github.com/dan-blanchard/7045057
if [[ $TRAVIS_OS_NAME == 'linux' ]]; then wget https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh -O miniconda.sh; fi
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then wget https://repo.continuum.io/miniconda/Miniconda3-4.2.12-MacOSX-x86_64.sh -O miniconda.sh; fi
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=$HOME/miniconda3/bin:$PATH
# Update conda itself
conda update --yes --quiet conda
PYTHON_VERSION_MAJOR=${TRAVIS_PYTHON_VERSION:0:1}
ENV_NAME="${TRAVIS_OS_NAME}_${CC}_${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}"
if [[ $PYTHON_VERSION_MAJOR == '2' ]]; then conda create --quiet --yes -n $ENV_NAME python=$TRAVIS_PYTHON_VERSION pip Cython=0.22 numpy=1.10.1 scipy nose matplotlib pandas; fi
if [[ $PYTHON_VERSION_MAJOR == '3' ]]; then conda create --quiet --yes -n $ENV_NAME python=$TRAVIS_PYTHON_VERSION pip Cython numpy scipy nose matplotlib pandas; fi
source activate $ENV_NAME
python -c "import numpy"

# run quietly due to travis ci's log limit
python setup.py -q install
