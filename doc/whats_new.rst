.. _whats_new:

.. currentmodule:: pystan

============
 What's New
============

v2.18.0.0 (TBD)
========================
- Update Stan source to v2.18.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.18.0>`_)
- Fit method ``to_dataframe`` organizes draws into a Pandas DataFrame. Only works if Pandas is installed. Thanks to Liam Brannigan and Ari Hartikainen.
- Improved effective sample size calculation. Thanks to Aki Vehtari.
- Implemented ``pystan.diagnostics``. Thanks to @jjramsey.
- Several bug fixes.

v2.17.1.0 (16. Jan 2018)
========================
- Update Stan source to v2.17.1 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.17.1>`_) (bugfix release)
- Remove deprecation warnings (Thanks to Alexander Rudiuk)
- Drop testing of Python 3.4.

v2.17.0.0 (6. Oct 2017)
=======================
- Update Stan source to v2.17.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.17.0>`_),
- Added ``pystan.stansummary`` function. Patch by @ahartikainen.
- Marked ``pystan.stan`` as deprecated. It will be removed in version 3.0.  Please compile and use a
  Stan program as two separate steps.
- Reminder: Consider using Python 3.5 or higher. In a future release, Python 3.4 wheels will no longer be
  built.
- Reminder: Stan v2.16.0 is the final release which will not require a C++11 compatible compiler. Future
  releases will require a C++11 compatible compiler. (This does not affect most users.)

v2.16.0.0 (22. June 2017)
========================
- Update Stan source to v2.16.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.16.0>`_),
- Ari Hartikainen (Aalto University) @ahartikainen joins the Stan development team. 🎉
- Added ``pystan.lookup`` (contributed by Marco Inacio, @randommm)
- NOTE: Stan v2.16.0 is the final release which will not require a C++11 compatible compiler. Future
  releases will require a C+11 compatible compiler. The vast majority of users have a compatible
  compiler.

v2.15.0.1 (2. May 2017)
========================
- Python 2.7 compatibility fix (#332). Thanks to @monga for the report.

v2.15.0.0 (21. Apr 2017)
========================
- Update Stan source to v2.15.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.15.0>`_)
- Allow UTF-8 comments in Stan Program code. Thanks to @ahartikainen
- Expose constrain_pars method, thanks to Lars Mescheder.

V2.14.0.0 (1. Jan 2017)
=========================
- Update Stan source to v2.14.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.14.0>`_),
  includes important fix to the default sampling algorithm (NUTS). All users are encouraged to upgrade.
- Several documentation and minor bug fixes (thanks @ahartikainen, @jrings, @nesanders)
- New OpenPGP signing key for use with PyPI. Key fingerprint is C3542448245BEC68F43070E4CCB669D9761F0CAC.

V2.12.0.0 (15. Sept 2016)
=========================
- Update Stan source to v2.12.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.12.0>`_)
- #239 Fix bug in array indexing (thanks @stephen-hoover)
- #254 FIx off-by-one error in estimated sample size calculation

V2.11.0.0 (28. July 2016)
=========================
- Update Stan source to v2.11.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.11.0>`_)

V2.10.0.0 (18. July 2016)
=========================
- Update Stan source to v2.10.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.10.0>`_)
- Sampling in ``Fixed_param`` mode now works. Thanks to @luac for the fix and @axch for the original report.
- Detailed installation instructions from @chendaniely added to the documentation.

v2.9.0.0 (7. Jan 2016)
======================
- Update Stan source to v2.9.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.9.0>`_)
- Bugs fixed in _chains.pyx and model.py (thanks to @stephen-hoover, Paul Kernfeld)

v2.8.0.2 (6. Nov 2015)
======================
- Cython 0.22 or higher requirement included on PyPI

v2.8.0.1 (5. Nov 2015)
======================
- Python 3.5 support added
- Cython 0.22 or higher now required
- Compiler optimization (``-O2``) turned on for model compilation. This should increase sampling speed.
- Significant bug fixes (pickling, pars keyword)

v2.8.0.0 (1. Oct 2015)
======================
- Update Stan source to v2.8.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.8.0>`_)

v2.7.0.1 (22. August 2015)
==========================
-   Minor Cython 0.23.1 compatibility fixes
-   Bug preventing `mean_pars` from being recorded

v2.7.0.0 (21. July 2015)
=======================
-   Update Stan source to v2.7.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.7.0>`_)

v2.6.3.0 (21. Mar 2015)
=======================
- Update Stan source to v2.6.3 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.6.3>`_).

v2.6.0.0 (9. Feb 2015)
=======================
- Update Stan source to v2.6.0 (`release notes <https://github.com/stan-dev/stan/releases/tag/v2.6.0>`_).

v2.5.0.2 (30. Jan 2015)
=======================
- Fix bug in rdump (for >1 dimensional arrays)

v2.5.0.1 (14. Nov 2014)
=======================
- Support for pickling fit objects (experimental)
- Fix bug that occurs when printing fit summary

v2.5.0.0 (21. Oct 2014)
=======================
- Update Stan source to v2.5.0
- Fix several significant bugs in the ``extract`` method

v2.4.0.3 (9. Sept 2014)
=======================
- Performance improvements for the printed summary of a fit.

v2.4.0.2 (6. Sept 2014)
=======================
- Performance improvements for the ``extract`` method (5-10 times faster)
- Performance improvements for the printed summary of a fit. Printing
  a summary of a model with more than a hundred parameters is not recommended.
  Consider using ``extract`` and calculating summary statistics for the
  parameters of interest.

v2.4.0.1 (31. July 2014)
========================
- Sets LBFGS as default optimizer.
- Adds preliminary support for Python binary wheels on OS X and Windows.
- Fixes bug in edge case in new summary code.

v2.4.0.0 (26. July 2014)
========================
- Stan 2.4 (LBFGS optimizer added, Nesterov removed)
- Improve display of fit summaries

v2.3.0.0 (26. June 2014)
========================
- Stan 2.3 (includes user-defined functions, among other improvements).
- Optimizing returns a vector (array) by default instead of a dictionary.

v2.2.0.1 (30. April 2014)
=========================
- Add support for reading Stan's R dump files.
- Add support for specifying parameters of interest in ``stan``.
- Add Windows installation instructions. Thanks to @patricksnape.
- Lighten source distribution.

v2.2.0.0 (16. February 2014)
============================
- Updates Stan to v2.2.0.

v2.1.0.1 (27. January 2014)
===========================
- Implement model name obfuscation. Thanks to @karnold
- Improve documentation of StanFit objects

v2.1.0.0 (26. December 2013)
============================
- Updates Stan code to v2.1.0.

v2.0.1.3 (18. December 2013)
============================
- Sampling is parallel by default.
- ``grad_log_prob`` method of fit objects is available.

v2.0.1.2 (1. December 2013)
============================
- Improves setuptools support.
- Allows sampling chains in parallel using multiprocessing. See the ``n_jobs``
  parameter for ``stan()`` and the ``sampling`` method.
- Allows users to specify initial values for chains.

v2.0.1.1 (18. November 2013)
============================
- Clean up random_seed handling (Stephan Hoyer).
- Add fit methods get_seed, get_inits, and get_stancode.

v2.0.1.0 (24. October 2013)
============================

- Updated to Stan 2.0.1.
- Specifying ``sample_file`` now works as expected.

v2.0.0.1 (23. October 2013)
============================

- Stan ``array`` parameters are now handled correctly.
- Ancillary methods added to fit instances.
- Fixed bug that caused parameters in ``control`` dict to be ignored.

v2.0.0.0 (21. October 2013)
============================

- Stan source updated to to 2.0.0.
- PyStan version now mirrors Stan version.
- Rudimentary plot and traceplot methods have been added to fit instances.
- Warmup and sampling progress now visible.

v.0.2.2 (28. September 2013)
============================

- ``log_prob`` method available from StanFit instances.
- Estimated sample size and Rhat included in summary.

v.0.2.1 (17. September 2013)
============================

- ``StanModel`` instances can now be pickled.
- Adds basic support for saving output to ``sample_file``.

v.0.2.0 (25. August 2013)
=========================

- ``optimizing`` method working for scalar, vector, and matrix parameters
- stanfit objects now have ``summary`` and ``__str__`` methods à la RStan
- stan source updated to commit cc82d51d492d26f754fd56efe22a99191c80217b (July 26, 2013)
- IPython-relevant bug fixes

v.0.1.1 (19. July 2013)
=======================

- Support for Python 2.7 and Python 3.3
- ``stan`` and ``stanc`` working with common arguments
