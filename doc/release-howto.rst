This documentation is intended for use by pystan developers.

Release checklist
=================

- Update doc/whats_new.rst
- Update snippet in doc/getting_started.rst, i.e., "wget https://github.com/stan-dev/pystan/archive/0.2.0.zip"
- Set version in setup.py
- Set version in pystan/__init__.py
- Set ISRELEASED = True in setup.py
- ``git tag``
- Push tag to github ``git push --tags``
- Git branch ``master`` should be fast-forwarded to ``develop``
- Assemble source distribution, sign it, upload to PyPI:

::

    python setup.py sdist
    gpg --detach-sign -a dist/pystan-2.0.1.2.tar.gz
    twine upload dist/*

- Update docs in readthedocs

    make html
    cp _build/html /tmp/html

After release
=============
- Add placeholder for changes in doc/whats_new.rst
- Set ISRELEASED = False in setup.py
- Bump version in setup.py
- Bump version in pystan/__init__.py

See also
========
- http://docs.astropy.org/en/v0.2/development/building_packaging.html
