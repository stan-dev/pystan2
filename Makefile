NOSETESTS ?= nosetests

default: test

build: pystan/_api.pyx pystan/_chains.pyx
	python setup.py build_ext --inplace

test:
	$(NOSETESTS) -v --process-timeout=360 --processes=-1 -w /tmp pystan.tests

clean:
	rm -rf pystan/*.so pystan/bin/libstan.a

check:
	g++ pystan/stan_fit.hpp -I pystan/stan/src/ -I pystan/stan/lib/boost_1.54.0/ -I pystan/stan/lib/eigen_3.2.0/
