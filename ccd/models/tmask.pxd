import numpy as np
cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t STYPE_t
ctypedef np.int16_t   DTYPE_t
ctypedef float        FTYPE_t
ctypedef int          ITYPE_t
ctypedef bool         BTYPE_t
ctypedef np.long_t    LTYPE_t


#cdef np.ndarray[STYPE_t, ndim=2] tmask_coefficient_matrix(np.ndarray[LTYPE_t, ndim=1] dates,
#                                                          FTYPE_t avg_days_yr)
cdef np.ndarray tmask_coefficient_matrix(np.ndarray[LTYPE_t, ndim=1] dates,
                                                          FTYPE_t avg_days_yr)



cpdef np.ndarray tmask(np.ndarray[LTYPE_t, ndim=1] dates,
                      np.ndarray[DTYPE_t, ndim=2] observations,
                      np.ndarray[STYPE_t, ndim=1] variogram,
                      list bands,
                      FTYPE_t t_const,
                      FTYPE_t avg_days_yr)