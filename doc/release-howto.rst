==================================
 How to make a release of PyStan
==================================

*The audience for this page is PyStan developers.*

- Update release notes, ``doc/whats_new.rst``.
- Double check version number in ``pystan/__init__.py``
- Update version in snippet in ``doc/getting_started.rst``, i.e., "wget
  https://github.com/stan-dev/pystan/archive/0.2.0.zip" (Note there are three
  references to the version number.)
- Commit any changes you just made.
- Verify building source distribution works, ``python setup.py sdist``
- Fast-forward branch ``master`` to ``develop``.
- Release documentation

  - ``git rebase master readthedocs``
  - ``git push --force origin readthedocs``

- Tag and sign the release

  - For example, ``git tag --sign v2.4.0.1``
  - Push tag to github ``git push --tags``

- Assemble source distribution, sign it, upload to PyPI::

    python setup.py sdist
    twine upload --sign dist/*

- Build OS X wheels
  
  - See https://github.com/ariddell/pystan-wheel-builder for instructions.
  - Download, sign, and upload wheels::

    twine upload --sign *.whl

After release
=============

- Checkout the ``develop`` branch
- Add placeholder for changes in ``doc/whats_new.rst``
- Update version in ``pystan/__init__.py``

See also
========
- http://docs.astropy.org/en/v0.2/development/building_packaging.html
- https://github.com/stefanv/scikit-image/blob/master/RELEASE.txt
