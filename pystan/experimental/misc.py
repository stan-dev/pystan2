import logging
import re
import os

logger = logging.getLogger('pystan')

def fix_include(model_code):
    """Function to normalize (remove whitespace) around the #include statements.
    
    Note
    ----
    This functions return edited version of the model_code.

    Parameters
    ----------
    model_code, str

    Returns
    -------
    new_model_code, str
    
    Example
    -------
    from pystan.experimental import fix_include
    model_code = fix_include(
    """
    pattern = r"(?<=\n)\s*(#include)\s*(\S+)\s*(?=\n)"
    model_code, n = re.subn(pattern, r"\1 \2", model_code)
    if n == 1:
        msg = "Made {} subsitution for the model_code"
    else:
        msg = "Made {} subsitutions for the model_code"
    logger.info(.format(n))
    return model_code
