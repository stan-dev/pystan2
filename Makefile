NOSETESTS ?= nosetests

default: test

build: pystan/stanfit4model.pyx
	python setup.py build_ext --inplace

test: build
	$(NOSETESTS) -v pystan

clean:
	rm -rf pystan/*.so
