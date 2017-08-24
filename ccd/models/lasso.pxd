import numpy as np
cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t STYPE_t
ctypedef float        FTYPE_t
ctypedef int          ITYPE_t
ctypedef bool         BTYPE_t
ctypedef np.long_t    LTYPE_t


cdef np.ndarray coefficient_matrix(np.ndarray[LTYPE_t, ndim=1],
                                   FTYPE_t,
                                   ITYPE_t)