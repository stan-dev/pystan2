This documentation is intended for use by pystan developers.

Release checklist
=================

- Double check version in ``pystan/__init__.py``
- Update ``doc/whats_new.rst``
- Bump version in snippet in ``doc/getting_started.rst``, i.e., "wget
  https://github.com/stan-dev/pystan/archive/0.2.0.zip" (Note there are three
  references to the version number.)
- Commit any changes you just made.
- Verify source distribution is working ``python setup.py sdist``
- ``git tag --sign``
- Push tag to github ``git push --tags``
- Git branch ``master`` should be fast-forwarded to ``develop``
- Release documentation:
  - ``git rebase master readthedocs``
  - ``git push --force origin readthedocs``
- Assemble source distribution, sign it, upload to PyPI:

::

    python setup.py sdist
    gpg --detach-sign -a dist/pystan-x.x.x.x.tar.gz
    twine upload dist/*

After release
=============

- Add placeholder for changes in ``doc/whats_new.rst``
- Bump version in ``pystan/__init__.py``

See also
========
- http://docs.astropy.org/en/v0.2/development/building_packaging.html
