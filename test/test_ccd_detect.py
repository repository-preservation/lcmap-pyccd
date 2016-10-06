""" Tests for running ccd from the top level __init__.py/detect() """
from shared import read_data
from shared import two_change_data

import ccd


def test_sample2_ccd_detect():
    """ Sample 2 contains two changes and should test the full path through
    the algorithm including preprocessing """
    data = read_data("test/resources/sample_2.csv")
    results = ccd.detect(*data)
    assert len(results) == 2


def test_detect_two_changes():
    """ This uses generated test data that contains two changes but fails
    during preprocessing due to nonsensical qa values.
    Thus preprocess==False """
    data = two_change_data()
    results = ccd.detect(*data, preprocess=False)
    assert len(results) == 2
