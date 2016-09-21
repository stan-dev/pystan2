==================================
 How to make a release of PyStan
==================================

*The audience for this page is PyStan developers.*

- Update release notes, ``doc/whats_new.rst``.
- Double check version number in ``pystan/__init__.py``
- Update version in snippet in ``doc/getting_started.rst``, i.e., "wget ..."
- Commit any changes you just made.
- Double check that tests pass for this commit.
- Fast-forward branch ``master`` to ``develop``.
- Release documentation

  - ``git rebase master readthedocs``
  - ``git push --force origin readthedocs``

- Tag and sign the release

  - For example, ``git tag --sign v2.4.0.1``
  - Push tag to github ``git push --tags``

- Assemble source distribution::

  ./build_dist.sh

- Upload source distribution::

    twine upload --sign dist/*

Build Wheels
------------

Linux and OSX: in the ``pystan-wheels`` repo update the ``pystan`` submodule
and bump the version in ``.travis.yml``. Push changes.

Windows: ``appveyor.yml`` in the ``pystan`` repo takes care of building Windows
wheels.

Upload Wheels
---------------

Use ``continuous_integration/upload_wheels.sh`` to download wheels and then
upload the wheels to PyPI.

After release
=============

- Checkout the ``develop`` branch
- Update version in ``pystan/__init__.py``
