"""
Tests for running ccd from the top level __init__.py/detect()

Sanity checks to make sure test data sets run to completion
"""
import numpy as np

from test.shared import read_data
# from shared import two_change_data
#
import ccd


def test_sample_data_sets():
    """
    Sanity test to ensure all test data sets run to completion
    """
    samples = ['test/resources/sample_1.csv',
               'test/resources/sample_2.csv',
               'test/resources/sample_WA_grid08_row9_col2267_persistent_snow.csv',
               'test/resources/sample_WA_grid08_row12_col2265_fmask_fail.csv',
               'test/resources/sample_WA_grid08_row999_col1_normal.csv',
               'test/resources/test_3657_3610_observations.csv']

    for sample in samples:
        data = read_data(sample)
        results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
                             data[5], data[6], data[7], data[8])


def test_sort_dates():
    arr = [1, 3, 2, 5]
    ans = np.array([0, 2, 1, 3])

    assert np.array_equal(ans, ccd.__sort_dates(arr))


def test_unique():
    arr = np.array([1, 3, 2, 5, 3, 2])
    ans = np.array([0, 2, 1, 3])

    assert np.array_equal(ans, ccd.__unique_indices(arr))
