# cython: profile=True
import numpy as np
cimport numpy as np

from cpython cimport bool

from ccd.models import FittedModel
from ccd.math_utils import calc_rmse

ctypedef np.float64_t STYPE_t
ctypedef float        FTYPE_t
ctypedef int          ITYPE_t
ctypedef bool         BTYPE_t
ctypedef np.long_t    LTYPE_t


cdef np.ndarray coefficient_matrix(np.ndarray[LTYPE_t, ndim=1] dates,
                                   FTYPE_t avg_days_yr,
                                   ITYPE_t num_coefficients):
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
    matrix = np.zeros(shape=(len(dates), 7), order='F')
    w12 = w * dates

    cos = np.cos
    sin = np.sin

    matrix[:, 0] = dates
    matrix[:, 1] = cos(w12)
    matrix[:, 2] = sin(w12)

    if num_coefficients >= 6:
        w34 = 2 * w12
        matrix[:, 3] = cos(w34)
        matrix[:, 4] = sin(w34)

    if num_coefficients >= 8:
        w56 = 3 * w12
        matrix[:, 5] = cos(w56)
        matrix[:, 6] = sin(w56)

    return matrix


cpdef fitted_model(np.ndarray[LTYPE_t, ndim=1] dates,
                   np.ndarray[STYPE_t, ndim=1] spectra_obs,
                   ITYPE_t max_iter,
                   FTYPE_t avg_days_yr,
                   ITYPE_t num_coefficients,
                   object lm):

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
    #print("***  spectra_obs: {} {}".format(spectra_obs.ndim, spectra_obs.dtype))
    coef_matrix = coefficient_matrix(dates, avg_days_yr, num_coefficients)
    model = lm.fit(coef_matrix, spectra_obs)
    #model = ElasticNet().fit(coef_matrix, spectra_obs)
    #lasso = linear_model.Lasso(max_iter=max_iter)
    #model = lasso.fit(coef_matrix, spectra_obs)

    predictions = model.predict(coef_matrix)
    rmse, residuals = calc_rmse(spectra_obs, predictions)

    return FittedModel(fitted_model=model, rmse=rmse, residual=residuals)


cpdef predict(object model,
              np.ndarray[LTYPE_t, ndim=1] dates,
              FTYPE_t avg_days_yr):
    coef_matrix = coefficient_matrix(dates, avg_days_yr, 8)

    return model.fitted_model.predict(coef_matrix)
