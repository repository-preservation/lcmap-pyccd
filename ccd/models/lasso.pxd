import numpy as np
cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t STYPE_t
ctypedef float        FTYPE_t
ctypedef int          ITYPE_t
ctypedef bool         BTYPE_t
ctypedef np.long_t    LTYPE_t


cdef np.ndarray coefficient_matrix(np.ndarray[LTYPE_t, ndim=1] dates,
                                   FTYPE_t avg_days_yr,
                                   ITYPE_t num_coefficients)

cpdef object fitted_model(np.ndarray[LTYPE_t, ndim=1] dates,
                          np.ndarray[STYPE_t, ndim=1] spectra_obs,
                          ITYPE_t max_iter,
                          FTYPE_t avg_days_yr,
                          ITYPE_t num_coefficients,
                          object lm)

cpdef predict(object model,
              np.ndarray[LTYPE_t, ndim=1] dates,
              FTYPE_t avg_days_yr)