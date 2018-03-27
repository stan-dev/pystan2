import logging
import re
import os

logger = logging.getLogger('pystan')

def fix_include(model_code):
    """

    Parameters
    ----------
    model_code, str

    Returns
    -------
    new_model_code, str
    """
    pattern = r"(?<=\n)\s*(#include)\s*(\S+)\s*(?=\n)"
    model_code, n = re.subn(pattern, r"\1 \2", model_code)
    logger.info("Made {} subsitutions for the model_code".format(n) 
    return model_code
