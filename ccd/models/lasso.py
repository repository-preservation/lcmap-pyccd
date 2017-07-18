from sklearn import linear_model
import numpy as np
from cachetools import LRUCache

from ccd.models import FittedModel
from ccd.math_utils import calc_rmse

import numba

cache = LRUCache(maxsize=1000)


@numba.jit(nopython=True, nogil=True, cache=True)
def __coefficient_cache_key(observation_dates):
    return tuple(observation_dates)


# @cached(cache=cache, key=__coefficient_cache_key)
@numba.jit(cache=True)
def coefficient_matrix(dates, avg_days_yr, num_coefficients):
    """
    Fourier transform function to be used for the matrix of inputs for
    model fitting

    Args:
        dates: list of ordinal dates
        num_coefficients: how many coefficients to use to build the matrix

    Returns:
        Populated numpy array with coefficient values
    """
    w = 2 * np.pi / avg_days_yr

    # http://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#other-functions
    # order kwarg not supported by numba, need to verify
    #matrix = np.zeros(shape=(len(dates), 7), order='F')
    matrix = np.zeros(shape=(len(dates), 7))

    matrix[:, 0] = dates
    matrix[:, 1] = np.cos(w * dates)
    matrix[:, 2] = np.sin(w * dates)

    if num_coefficients >= 6:
        matrix[:, 3] = np.cos(2 * w * dates)
        matrix[:, 4] = np.sin(2 * w * dates)

    if num_coefficients >= 8:
        matrix[:, 5] = np.cos(3 * w * dates)
        matrix[:, 6] = np.sin(3 * w * dates)

    return matrix


@numba.jit(cache=True)
def fitted_model(dates, spectra_obs, max_iter, avg_days_yr, num_coefficients):
    """Create a fully fitted lasso model.

    Args:
        dates: list of ordinal observation dates
        spectra_obs: list of values corresponding to the observation dates for
            a single spectral band
        num_coefficients: how many coefficients to use for the fit
        max_iter: maximum number of iterations that the coefficients
            undergo to find the convergence point.

    Returns:
        sklearn.linear_model.Lasso().fit(observation_dates, observations)

    Example:
        fitted_model(dates, obs).predict(...)
    """
    coef_matrix = coefficient_matrix(dates, avg_days_yr, num_coefficients)

    lasso = linear_model.Lasso(max_iter=max_iter)
    model = lasso.fit(coef_matrix, spectra_obs)

    predictions = model.predict(coef_matrix)
    rmse, residuals = calc_rmse(spectra_obs, predictions)

    #return FittedModel(fitted_model=model, rmse=rmse, residual=residuals)
    return {'fitted_model': model, 'rmse': rmse, 'residual': residuals}


@numba.jit(cache=True)
def predict(model, dates, avg_days_yr):
    coef_matrix = coefficient_matrix(dates, avg_days_yr, 8)
    #return model.fitted_model.predict(coef_matrix)
    return model['fitted_model'].predict(coef_matrix)