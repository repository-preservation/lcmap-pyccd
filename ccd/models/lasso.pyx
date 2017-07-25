from sklearn import linear_model
import numpy as np
cimport numpy as np

from ccd.models import FittedModel
#from ccd.math_utils import calc_rmse


#ITYPE = np.int
#FTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
#ctypedef np.int_t ITYPE_t
#ctypedef np.float_t FTYPE_t

#def __coefficient_cache_key(observation_dates):
#    return tuple(observation_dates)


cdef np.ndarray calc_residuals(np.ndarray actual,
                   np.ndarray predicted):
    """
    Helper method to make other code portions clearer

    Args:
        actual: 1-d array of observed values
        predicted: 1-d array of predicted values

    Returns:
        ndarray: 1-d array of residual values
    """
    return actual - predicted


cdef tuple calc_rmse(np.ndarray actual, np.ndarray predicted):
    """
    Calculate the root mean square of error for the given inputs

    Args:
        actual: 1-d array of values, observed
        predicted: 1-d array of values, predicted

    Returns:
        float: root mean square value
        1-d ndarray: residuals
    """
    cdef np.ndarray residuals = calc_residuals(actual, predicted)

    return (residuals ** 2).mean() ** 0.5, residuals


cdef np.ndarray coefficient_matrix(np.ndarray dates,
                                    np.float avg_days_yr,
                                    np.int num_coefficients):
    """
    Fourier transform function to be used for the matrix of inputs for
    model fitting

    Args:
        dates: list of ordinal dates
        num_coefficients: how many coefficients to use to build the matrix

    Returns:
        Populated numpy array with coefficient values
    """
    #w = 2 * np.pi / avg_days_yr
    cdef float w = 2 * np.pi / avg_days_yr

    #matrix = np.zeros(shape=(len(dates), 7), order='F')
    cdef int msize = 7
    cdef int dsize = len(dates)
    cdef np.ndarray matrix = np.zeros(shape=(dsize, msize), dtype=np.float)

    cdef np.ndarray dcos = np.cos(w * dates)
    cdef np.ndarray dsin = np.sin(w * dates)

    matrix[:, 0] = dates
    matrix[:, 1] = dcos
    matrix[:, 2] = dsin

    if num_coefficients >= 6:
        matrix[:, 3] = np.cos(2 * w * dates)
        matrix[:, 4] = np.sin(2 * w * dates)

    if num_coefficients >= 8:
        matrix[:, 5] = np.cos(3 * w * dates)
        matrix[:, 6] = np.sin(3 * w * dates)

    return matrix


cpdef fitted_model(np.ndarray dates,
                  np.ndarray spectra_obs,
                  np.int max_iter,
                  np.float avg_days_yr,
                  np.int num_coefficients):


    """Create a fully fitted lasso model.

    Args:
        dates: list or ordinal observation dates
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
    # change

    cdef np.ndarray coef_matrix = coefficient_matrix(dates, avg_days_yr, num_coefficients)

    lasso = linear_model.Lasso(max_iter=max_iter)
    model = lasso.fit(coef_matrix, spectra_obs)

    predictions = model.predict(coef_matrix)
    rmse, residuals = calc_rmse(spectra_obs, predictions)

    return FittedModel(fitted_model=model, rmse=rmse, residual=residuals)

cpdef predict(model, dates, avg_days_yr):
    coef_matrix = coefficient_matrix(dates, avg_days_yr, 8)

    return model.fitted_model.predict(coef_matrix)
