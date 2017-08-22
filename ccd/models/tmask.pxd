import numpy as np
cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t STYPE_t
ctypedef float        FTYPE_t
ctypedef int          ITYPE_t
ctypedef bool         BTYPE_t
ctypedef np.long_t    LTYPE_t


cdef np.ndarray tmask(np.ndarray[LTYPE_t, ndim=1],
                      np.ndarray[STYPE_t, ndim=2],
                      np.ndarray[STYPE_t, ndim=1],
                      list,
                      FTYPE_t,
                      FTYPE_t)