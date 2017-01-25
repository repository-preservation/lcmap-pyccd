"""
Contains commonly used math functions.

This file is meant to help code reuse, profiling, and look at speeding up
individual operations.

In the interest of avoiding circular imports, this should be kept to be fairly
stand-alone. I.e. it should not import any other piece of the overall project.
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
    def f(*args, **kwargs):
        return func(*(np.asarray(_) for _ in args), **kwargs)
    return f


@ensure_ndarray_input
def adjusted_variogram(vector):
    """
    Calculate a modified first order variogram/madogram

    Args:
        vector: 1-d array of values

    Returns:
        float
    """
    return np.median(np.abs(np.diff(vector)))


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
def euclidean_norm_sq(vector, axis=None):
    """
    Return the square of the euclidean norm, essentially removes
    the square root

    This is used to simplify some math used during
    change detection processing

    Args:
        vector: 1-d array of values
        axis: axis along which to perform the summation process

    Returns:
        float
    """
    return np.sum(vector ** 2, axis=axis)


@ensure_ndarray_input
def sum_of_squares(vector, axis=None):
    return np.sum(vector ** 2, axis=axis)


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
def calc_median(vector):
    """
    Calculate the median value of the given vector

    Args:
        vector: array of values

    Returns:
        float: median value
    """
    return np.median(vector)


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

    L2 processing for the thermal band (known as Brightness Temperature) is
    initially done in kelvin and has been scaled by a factor of 10 already,
    in the interest of keeping the values in integer space, a further factor
    of 10 is calculated.

    scaled C = K * 10 - 27315
    unscaled C = K / 10 - 273.15

    Args:
        thermals: 1-d ndarray of scaled thermal values in kelvin
        scale: int scale factor used for the thermal values

    Returns:
        1-d ndarray of thermal values in scaled degrees celsius
    """
    return thermals * scale - 27315


@ensure_ndarray_input
def calculate_variogram(observations):
    """
    Calculate the first order variogram/madogram across all bands

    Helper method to make subsequent code clearer

    Args:
        observations: spectral band values

    Returns:
        1-d ndarray representing the variogram values
    """
    return np.median(np.abs(np.diff(observations)), axis=1)



