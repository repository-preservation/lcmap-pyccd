"""
Tests for running ccd from the top level __init__.py/detect()

Sanity checks to make sure test data sets run to completion
"""
import numpy as np
import pandas as pd

from test.shared import read_data
from test.shared import rainbow
from test.shared import TEST_UBIDS
from test.shared import dtstr_to_ordinal
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

    params = {'QA_BITPACKED': False,
              'QA_FILL': 255,
              'QA_CLEAR': 0,
              'QA_WATER': 1,
              'QA_SHADOW': 2,
              'QA_SNOW': 3,
              'QA_CLOUD': 4}

    for sample in samples:
        data = read_data(sample)
        results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
                             data[5], data[6], data[7], data[8],
                             params=params)


def test_real_data():
    """
    Test with data available locally as json
    """
    row, col = 57, 97
    rbow = rainbow(-2094585, 1682805, '1982-01-01/2015-12-31', "chipsurl", "specsurl", TEST_UBIDS)
    rainbow_date_array = np.array(rbow['t'].values)

    ccd.detect([dtstr_to_ordinal(str(pd.to_datetime(i)), False) for i in rainbow_date_array],
               np.array(rbow['blue'].values[:, row, col]),
               np.array(rbow['green'].values[:, row, col]),
               np.array(rbow['red'].values[:, row, col]),
               np.array(rbow['nir'].values[:, row, col]),
               np.array(rbow['swir1'].values[:, row, col]),
               np.array(rbow['swir2'].values[:, row, col]),
               np.array(rbow['thermal'].values[:, row, col]),
               np.array(rbow['cfmask'].values[:, row, col], dtype=int),
               params={})


def test_sort_dates():
    arr = [1, 3, 2, 5, 2]
    ans = np.array([0, 2, 4, 1, 3])

    assert np.array_equal(ans, ccd.__sort_dates(arr))
