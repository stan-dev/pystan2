==================================
 How to make a release of PyStan
==================================

*The audience for this page is PyStan Developers.*

*The release signing key for PyStan was created on 2017-01-01 and has
fingerprint C3542448245BEC68F43070E4CCB669D9761F0CAC.*

All steps except those described in "Tag Release" should be performed on a
server with a high-bandwidth connection to the Internet. About 1000MiB worth of
data will be uploaded to PyPi.

Before Tagging Release
======================
- Update release notes, ``doc/whats_new.rst``.
- Double check version number in ``pystan/__init__.py``
- Commit any changes you just made.
- Double check that tests pass for this commit.

Tag Release
===========

- Fast-forward branch ``master`` to ``develop``.
- Tag and sign the release

  - For example, ``git tag -u C3542448245BEC68F43070E4CCB669D9761F0CAC --sign v2.4.0.1``
  - Push tag to github ``git push --tags``

Update Documentation
=====================

*readthedocs needs a single patch to work.*

- Release documentation

  - ``git rebase master readthedocs``
  - ``git push --force origin readthedocs``

- Manually trigger a build on the readthedocs website.

Build source distribution and wheels
====================================

Assemble source distribution::

    ./build_dist.sh

Linux and OSX: in the ``pystan-wheels`` repo update the ``pystan`` submodule
and bump the version in ``.travis.yml``. Push changes.

Windows: ``appveyor.yml`` in the ``pystan`` repo takes care of building Windows
wheels.

When wheels have been created they will be automatically uploaded to a
Rackspace storage bucket.

After the wheels have finished building, download them from the Rackspace
storage bucket.

Use ``continuous_integration/download_wheels.sh`` to download all wheels into
the directory ``dist/``.

Upload Source Distribution and Wheels to PyPI
=============================================

- Upload source distribution and wheels::

    python3 -m twine upload --skip-existing dist/*.tar.gz dist/*.whl

If ``twine`` prompts for a username and password abort the process with
Control-C and enter your PyPI credentials in ``$HOME/.pypirc``. (For more
details see the Python documention on `the .pypirc file
<https://docs.python.org/3/distutils/packageindex.html#pypirc>`_.) Alternatively,
one can set the environment variables ``TWINE_USERNAME`` and ``TWINE_PASSWORD``.

Uploading wheels may take a long time on a low-bandwidth connection.

Post-release Tasks
==================

Update Source
-------------

- Checkout the ``develop`` branch.
- Update version in ``pystan/__init__.py`` to ``<n.n.n.n+1>dev``.
- Add placeholder for next release in ``doc/whats_new.rst``.
- Commit changes and push ``develop``.

Update Conda-Forge
------------------

Update the repository at https://github.com/conda-forge/pystan-feedstock by
editing ``recipe/meta.yaml`` and submitting a pull request.

You will need the sha256 of the tarball. Calculate it by hand (with ``sha256sum``) or find it at
https://pypi.org/project/pystan/#files after uploading the tarball.

Update and Tag CVODES branch
----------------------------
- Checkout the ``cvodes`` branch.
- Rebase this on the current release.
- Tag this current commit with a tag that includes the suffix ``-cvodes`` (e.g., `v2.18.0.0-cvodes`).
- Push the tag.

Make Release Announcement
-------------------------

Post a message to the Stan discourse forum. The following is an example from
PyStan 2.19.1.1::

    PyStan 2.19.1.1 is now available on PyPI. You may upgrade with

         python3 -m pip install --upgrade pystan

    A list of changes is available in the customary location:

    http://pystan2.readthedocs.io/en/latest/whats_new.html

TODO
====

- Automate more of this process.
