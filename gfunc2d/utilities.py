import re
import numpy as np


def is_color(color_str):
    '''
    Check if a string is the name of a color index like for example 'B-V'.
    Any two groups of letters and numbers separated by a single dash will
    pass as a color index.

    Parameters
    ----------
    color_str : str
        The string to be tested.

    Returns
    -------
    result : bool
        True if the input string is a color index.
    '''

    regex_match = re.fullmatch('([a-zA-Z0-9]+)-([a-zA-Z0-9]+)', color_str)

    if regex_match is None:
        result = False
    else:
        result = True

    return result


def find_nearest(array, value):
    '''
    Find the element of an array that is nearest some value and return it

    Parameters
    ----------
    array : array
        An array of values to search.

    value : float
        The value for which the nearest element should be returned.

    Returns
    -------
    nearest : float
        The array element which is closest to 'value'
    '''

    idx = np.argmin((np.abs(array-value)))
    nearest = array[idx]

    return nearest
