"""
Tests for any methods that reside in the ccd.models sub-module.
"""

import numpy as np

from test.shared import read_data

from ccd import models


def test_lasso_coefficient_matrix():
    sample = 'test/resources/sample_WA_grid08_row999_col1_normal.csv'
    avg_days_yr = 365.25
    num_coefficients = (4, 6, 8)

    dates = read_data(sample)[0]

    for coefs in num_coefficients:
        matrix = models.lasso.coefficient_matrix(dates, avg_days_yr, coefs)
        col_idx = coefs - 1

        used_cols = matrix[:, :col_idx]
        unused_cols = matrix[:, col_idx:]
        print(matrix.shape)
        print(used_cols.shape)
        print(unused_cols.shape)
        print(len(dates))

        # Ensure that we are covering the entire matrix array.
        assert used_cols.shape[1] + unused_cols.shape[1] == matrix.shape[1]

        # Check to make sure we have our matrix columns filled out to a point.
        assert np.where(used_cols == 0)[0].size == 0

        # And make sure the rest is zero'd out.
        if unused_cols.shape[1] > 0:
            assert (np.where(unused_cols == 0)[0].size / unused_cols.shape[1])\
                   == len(dates)
