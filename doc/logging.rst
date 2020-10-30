.. _logging:

.. currentmodule:: pystan

=======
Logging
=======

PyStan uses logging-module from the Python Standard library to output messages for the user. 
By default, messages are sent to ``sys.stdout``. For more information and usage see `logging-module <https://docs.python.org/3.7/library/logging.handlers.html>`_ and `logging-cookbook <https://docs.python.org/3/howto/logging-cookbook.html>`_

Advanced users
==============

To add other logger handlers or to redirect all messages from PyStan, the user needs to setup output handlers manually.
If the setup is done before importing PyStan, PyStan won't add automatically ``logging.StreamHandler`` to logging handlers.
Otherwise, PyStan adds ``logging.StreamHandler`` and other handlers coexist with the default handler.

Adding FileHandler
------------------

To redirect all messages only to a file.

.. code-block:: python

    import logging
    logger = logging.getLogger("pystan")
    
    # add root logger (logger Level always Warning)
    # not needed if PyStan already imported
    logger.addHandler(logging.NullHandler()) 
    
    logger_path = "pystan.log"
    fh = logging.FileHandler(logger_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    # optional step
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    import pystan

To use both (default, file) logging options import pystan before the setup.
In this case PyStan adds root handler, which means that user can skip the root handler step.

.. code-block:: python

    import pystan
    
    import logging
    logger = logging.getLogger("pystan")
    ...

