import numpy as np
cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t ITYPE_t
ctypedef float FTYPE_t
ctypedef np.complex_t CTYPE_t

cpdef np.ndarray[ITYPE_t, ndim=1] bisquare(np.ndarray[ITYPE_t, ndim=1] resid, FTYPE_t c=4.685):
    """
    Returns weighting for each residual using bisquare weight function

    Args:
        resid (np.ndarray): residuals to be weighted
        c (float): tuning constant for Tukey's Biweight (default: 4.685)

    Returns:
        weight (ndarray): weights for residuals

    Reference:
        http://statsmodels.sourceforge.net/stable/generated/statsmodels.robust.norms.TukeyBiweight.html
    """
    # Weight where abs(resid) < c; otherwise 0
    cdef np.ndarray[ITYPE_t, ndim=1] abs_resid = np.abs(resid)
    return (abs_resid < c) * (1 - (resid / c) ** 2) ** 2

cpdef ITYPE_t mad(np.ndarray[ITYPE_t, ndim=1] x, ITYPE_t c=0.6745):
    """
    Returns Median-Absolute-Deviation (MAD) of some data

    Args:
        resid (np.ndarray): Observations (e.g., residuals)
        c (float): scale factor to get to ~standard normal (default: 0.6745)
                 (i.e. 1 / 0.75iCDF ~= 1.4826 = 1 / 0.6745)

    Returns:
        float: MAD 'robust' standard deivation  estimate

    Reference:
        http://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    # Return median absolute deviation adjusted sigma
    rs = np.sort(np.abs(x))
    return np.median(rs[4:]) / c

cpdef bool _check_converge(np.ndarray[ITYPE_t, ndim=1] x0, np.ndarray[ITYPE_t, ndim=1] x, ITYPE_t tol=1e-8):
    return not np.any(np.fabs(x0 - x > tol))

cpdef np.ndarray[ITYPE_t, ndim=1] _weight_beta(np.ndarray[ITYPE_t, ndim=2] X,
                                               np.ndarray[ITYPE_t, ndim=1] y,
                                               np.ndarray[ITYPE_t, ndim=1] w):

    cdef np.ndarray[ITYPE_t, ndim=1] sw = np.sqrt(w)

    cdef np.ndarray[ITYPE_t, ndim=2] Xw = X * sw[:, None]
    cdef np.ndarray[ITYPE_t, ndim=1] yw = y * sw

    return np.linalg.lstsq(Xw, yw)[0]

cpdef np.ndarray[ITYPE_t, ndim=1] _weight_resid(np.ndarray[ITYPE_t, ndim=2] X,
                                                np.ndarray[ITYPE_t, ndim=1] y,
                                                np.ndarray[ITYPE_t, ndim=1] beta):
    return y - np.dot(X, beta)



