"""
Tests for running ccd from the top level __init__.py/detect()

Sanity checks to make sure test data sets run to completion
"""
import numpy as np

from test.shared import read_data
# from shared import two_change_data
#
import ccd

params = ccd.app.get_default_params()


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


def test_npy():
    """
    Sanity tests for npy test data sets
    """
    samples = ['test/resources/h03v09_-2010765_1964625_pixel.npy']  # Main loop failure in LF

    for sample in samples:
        dat = np.load(sample)
        results = ccd.detect(**dat[1])


def test_insuff_clear():
    """
    Flex the insufficient clear procedure code to make sure it runs.
    """
    sample = 'test/resources/h04v03_-1947075_2846265_pixel_insuff.npy'

    result = ccd.detect(**np.load(sample)[1])

    assert result['change_models'][0]['curve_qa'] == params.CURVE_QA['INSUF_CLEAR']


def test_perm_snow():
    """
    Flex the permanent snow procedure code.
    """
    sample = 'test/resources/h04v03_-1947105_2846265_pixel_snow.npy'

    result = ccd.detect(**np.load(sample)[1])

    assert result['change_models'][0]['curve_qa'] == params.CURVE_QA['PERSIST_SNOW']


def test_startfit():
    """
    Flex the code the generates start fits.
    """
    sample = 'test/resources/h04v03_-1945155_2844645_pixel_startfit.npy'

    result = ccd.detect(**np.load(sample)[1])

    assert result['change_models'][0]['curve_qa'] == params.CURVE_QA['START']


def test_endfit():
    """
    Flex the code the generates start fits.
    """
    sample = 'test/resources/h04v03_-1945125_2844645_pixel_endfit.npy'

    result = ccd.detect(**np.load(sample)[1])

    assert result['change_models'][-1]['curve_qa'] == params.CURVE_QA['END']


def test_sort_dates():
    arr = [1, 3, 2, 5, 2]
    ans = np.array([0, 2, 4, 1, 3])

    assert np.array_equal(ans, ccd.__sort_dates(arr))
