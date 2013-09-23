NOSETESTS ?= nosetests

default: test

build: pystan/_api.pyx pystan/_chains.pyx
	python setup.py build_ext --inplace

test:
	$(NOSETESTS) -v --process-timeout=360 --processes=-1 -w /tmp pystan

clean:
	rm -rf pystan/*.so pystan/bin/libstan.a
