NOSETESTS ?= nosetests

default: test

build: pystan/_api.pyx
	python setup.py build_ext --inplace

test:
	$(NOSETESTS) -v --process-timeout=360 --processes=-1 -w pystan/tests

clean:
	rm -rf pystan/*.so pystan/bin/libstan.a
