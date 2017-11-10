==================================
 How to make a release of PyStan
==================================

*The audience for this page is PyStan Developers.*

*The release signing key for PyStan was created on 2017-01-01 and has
fingerprint C3542448245BEC68F43070E4CCB669D9761F0CAC.*

All steps except those described in "Tag Release" should be performed on a
server with a high-bandwidth connection to the Internet. About 1000MiB worth of
data will be uploaded to PyPi.

Tag Release
===========

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

Build source distribution
=========================

Assemble source distribution::

    ./build_dist.sh

Build Wheels
============

Linux and OSX: in the ``pystan-wheels`` repo update the ``pystan`` submodule
and bump the version in ``.travis.yml``. Push changes.

Windows: ``appveyor.yml`` in the ``pystan`` repo takes care of building Windows
wheels.

When wheels have been created they will be automatically uploaded to a
Rackspace storage bucket.

Download Wheels
---------------

After the wheels have finished building, download them from the Rackspace
storage bucket.

Use ``continuous_integration/download_wheels.sh`` to download all wheels into
the directory ``dist/``.

Upload Source Distribution and Wheels to PyPI
=============================================

*NOTE: EXPERIMENTAL*

- Sign source distribution and wheels::

    for tarball in dist/*.tar.gz; do
        gpg --detach-sign -a -u C3542448245BEC68F43070E4CCB669D9761F0CAC "$tarball"
    done

    for whl in dist/*.whl; do
        gpg --detach-sign -a -u C3542448245BEC68F43070E4CCB669D9761F0CAC "$whl"
    done

- Upload source distribution and wheels::

    twine upload --skip-existing dist/*.tar.gz dist/*.tar.gz.asc
    twine upload --skip-existing dist/*.whl dist/*.whl.asc

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

Update Stan Website
-------------------

Update the Stan website with the new PyStan version information. The version
number in the following file needs to be incremented::

    https://github.com/stan-dev/stan-dev.github.io/blob/master/citations/index.md

Make Release Announcement
-------------------------

Post a message to the stan-users mailing list. The following is an example from
PyStan 2.14.0.0 (subject is "pystan 2.14.0.0 released on PyPI")::

    PyStan 2.14.0.0 is now available on PyPI. You may upgrade with

         python -m pip install --upgrade pystan

    A list of changes is available in the customary location:

    http://pystan.readthedocs.io/en/latest/whats_new.html


TODO
====

- Automate this entire process.
