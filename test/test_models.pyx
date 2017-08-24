"""
Cythonized module for pulling in any cdef'd functions
"""

import numpy as np
cimport numpy as np

from test.shared import read_data

from ccd import models
from ccd.models.lasso cimport coefficient_matrix


def coefficient_matrix_wrap(dates, ady, coefs):
    return coefficient_matrix(dates, ady, coefs)