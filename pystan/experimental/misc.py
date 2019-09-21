import logging
import os
import platform
import re

logger = logging.getLogger('pystan')

def fix_include(model_code):
    """Reformat `model_code` (remove whitespace) around the #include statements.

    Note
    ----
        A modified `model_code` is returned.

    Parameters
    ----------
    model_code : str
        Model code

    Returns
    -------
    str
        Reformatted model code

    Example
    -------

    >>> from pystan.experimental import fix_include
    >>> model_code = "parameters { #include myfile.stan \n..."
    >>> model_code_reformatted = fix_include(model_code)
    # "parameters {#include myfile.stan\n..."

    """
    pattern = r"(?<=\n)\s*(#include)\s*(\S+)\s*(?=\n)"
    model_code, n = re.subn(pattern, r"\1 \2", model_code)
    if n == 1:
        msg = "Made {} substitution for the model_code"
    else:
        msg = "Made {} substitutions for the model_code"
    logger.info(msg.format(n))
    return model_code


def windows_short_path(path: str) -> str:
    """
    Gets the short path name of a given long path.
    http://stackoverflow.com/a/23598461/200291
    On non-Windows platforms, returns the path
    If (base)path does not exist, function raises RuntimeError
    """
    if platform.system() != 'Windows':
        return path

    if os.path.isfile(path) or (
        not os.path.isdir(path) and os.path.splitext(path)[1] != ''
    ):
        base_path, file_name = os.path.split(path)
    else:
        base_path, file_name = path, ''

    if not os.path.exists(base_path):
        raise RuntimeError(
            'Windows short path function needs a valid directory. '
            'Base directory does not exist: "{}"'.format(base_path)
        )

    import ctypes
    from ctypes import wintypes

    _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    _GetShortPathNameW.restype = wintypes.DWORD

    output_buf_size = 0
    while True:
        output_buf = ctypes.create_unicode_buffer(output_buf_size)
        needed = _GetShortPathNameW(base_path, output_buf, output_buf_size)
        if output_buf_size >= needed:
            short_base_path = output_buf.value
            break
        else:
            output_buf_size = needed

    short_path = (
        os.path.join(short_base_path, file_name)
        if file_name
        else short_base_path
    )
    return short_path
