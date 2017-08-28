import numpy as np
cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t STYPE_t
ctypedef float        FTYPE_t
ctypedef int          ITYPE_t
ctypedef bool         BTYPE_t
ctypedef np.long_t    LTYPE_t


cpdef np.ndarray[STYPE_t, ndim=1] bisquare(np.ndarray[STYPE_t, ndim=1] resid,
                                          FTYPE_t c=*)

cpdef STYPE_t mad(np.ndarray[STYPE_t, ndim=1] x,
                 STYPE_t c=*)

cpdef bool _check_converge(np.ndarray[STYPE_t, ndim=1] x0,
                          np.ndarray[STYPE_t, ndim=1] x,
                          STYPE_t tol=*)

cpdef np.ndarray[STYPE_t, ndim=1] _weight_beta(np.ndarray[STYPE_t, ndim=2] X,
                                                 np.ndarray[STYPE_t, ndim=1] y,
                                                 np.ndarray[STYPE_t, ndim=1] w)

cpdef np.ndarray[STYPE_t, ndim=1] _weight_resid(np.ndarray[STYPE_t, ndim=2] X,
                                                  np.ndarray[STYPE_t, ndim=1] y,
                                                  np.ndarray[STYPE_t, ndim=1] beta)



cdef class RLM:
    cdef public FTYPE_t tune
    cdef public FTYPE_t scale_constant
    cdef public BTYPE_t update_scale
    cdef public ITYPE_t maxiter
    cdef public STYPE_t tol
    cdef public np.ndarray coef_
    cdef public STYPE_t intercept_
    cdef public STYPE_t scale
    cdef public np.ndarray weights

    cpdef object fit(self, np.ndarray X, np.ndarray y)

    cpdef np.ndarray predict(self, np.ndarray X)

