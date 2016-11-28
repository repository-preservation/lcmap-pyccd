"""
Contains commonly used math functions
this file is meant to help code reuse and profiling purposes
"""
from functools import wraps

import numpy as np


# TODO: Cache timings
# TODO: Numba timings


def ensure_ndarray_input(func):
    """
    Wrapper to ensure inputs to a method are of type ndarray
    This cleans up subsequent code that might need to check for this
    """
    @wraps(func)
    def f(*args):
        return func(*(np.asarray(_) for _ in args))
    return f


@ensure_ndarray_input
def euclidean_norm(vector):
    """
    Calculate the euclidean norm across a vector

    This is the default norm method used by Matlab

    Args:
        vector: 1-d array of values

    Returns:
        float
    """
    return np.sum(vector ** 2) ** .5


@ensure_ndarray_input
def euclidean_norm_sq(vector):
    """
    Return the square of the euclidean norm, essentially removes
    the square root

    This is used to simplify some math used during
    change detection processing

    Args:
        vector: 1-d array of values

    Returns:
        float
    """
    return np.sum(vector ** 2)


@ensure_ndarray_input
def calc_rmse(actual, predicted):
    """
    Calculate the root mean square of error for the given inputs

    Args:
        actual: 1-d array of values, observed
        predicted: 1-d array of values, predicted

    Returns:
        float: root mean square value
        1-d ndarray: residuals
    """
    residuals = calc_residuals(actual, predicted)

    return (residuals ** 2).mean() ** 0.5, residuals


@ensure_ndarray_input
def calc_residuals(actual, predicted):
    """
    Helper method to make other code portions clearer

    Args:
        actual: 1-d array of observed values
        predicted: 1-d array of predicted values

    Returns:
        ndarray: 1-d array of residual values
    """
    return actual - predicted


@ensure_ndarray_input
def kelvin_to_celsius(thermals, scale=10):
    """
    Convert kelvin values to celsius

    Args:
        thermals:
        scale:

    Returns:

    """
    return thermals * scale - 27315
