import numpy as np
cimport numpy as np

from cpython cimport bool

ctypedef np.float64_t STYPE_t
ctypedef float        FTYPE_t
ctypedef int          ITYPE_t
ctypedef bool         BTYPE_t
ctypedef np.long_t    LTYPE_t


cpdef standard_procedure(np.ndarray[LTYPE_t, ndim=1] dates,
                         np.ndarray[STYPE_t, ndim=2] observations,
                         object fitter_fn,
                         np.ndarray[LTYPE_t, ndim=1] quality,
                         dict proc_params)

cdef initialize(np.ndarray[LTYPE_t, ndim=1] dates,
                np.ndarray[STYPE_t, ndim=2] observations,
                object fitter_fn,
                slice model_window,
                np.ndarray processing_mask,
                np.ndarray[STYPE_t, ndim=1] variogram,
                dict proc_params,
                object lasso)

cdef lookforward(np.ndarray[LTYPE_t, ndim=1] dates,
                 np.ndarray[STYPE_t, ndim=2] observations,
                 slice model_window,
                 object fitter_fn,
                 np.ndarray processing_mask,
                 np.ndarray[STYPE_t, ndim=1] variogram,
                 dict proc_params,
                 object lasso)

cdef lookback(np.ndarray[LTYPE_t, ndim=1] dates,
              np.ndarray[STYPE_t, ndim=2] observations,
              slice model_window,
              list models,
              ITYPE_t previous_break,
              np.ndarray processing_mask,
              np.ndarray[STYPE_t, ndim=1] variogram,
              dict proc_params)

cdef catch(np.ndarray[LTYPE_t, ndim=1] dates,
           np.ndarray[STYPE_t, ndim=2] observations,
           object fitter_fn,
           np.ndarray processing_mask,
           slice model_window,
           ITYPE_t curve_qa,
           dict proc_params,
           object lasso)








