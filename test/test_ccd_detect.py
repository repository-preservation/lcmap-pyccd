""" Tests for running ccd from the top level __init__.py/detect() """
from shared import read_data
from shared import two_change_data

import ccd


def test_validate_no_preprocessing_sample_1_detection_results():
    """Sample 1 contains one change and should test the full path through
    the algorithm without preprocessing.
    """
    data = read_data("test/resources/sample_1.csv")
    results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
                         data[5], data[6], data[7], data[8], preprocess=False)
    assert len(results) != 1, "expected: !{}, actual: {}".format(1, len(results))


def test_validate_sample_1_detection_results():
    """Sample 1 contains one change and should test the full path through
    the algorithm including preprocessing.
    """
    data = read_data("test/resources/sample_1.csv")
    results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
                         data[5], data[6], data[7], data[8], preprocess=True)
    assert len(results) == 1, "expected: {}, actual: {}".format(1, len(results))


def test_validate_no_preprocessing_sample_2_detection_results():
    """ Sample 2 contains two changes and should test the full path through
    the algorithm including preprocessing """
    data = read_data("test/resources/sample_2.csv")
    results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
                         data[5], data[6], data[7], data[8], preprocess=False)
    assert len(results) != 2, "expected: !{}, actual: {}".format(2, len(results))


def test_validate_sample_2_detection_results():
    """Sample 2 contains two changes and should test the full path through
    the algorithm including preprocessing"""
    data = read_data("test/resources/sample_1.csv")
    results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
                         data[5], data[6], data[7], data[8], preprocess=True)
    assert len(results) == 2, "expected: {}, actual: {}".format(2, len(results))


def test_two_changes_in_generated_sinusoidal_data():
    """ This uses generated test data that contains two changes but fails
    during preprocessing due to nonsensical qa values.
    Thus preprocess==False """
    data = two_change_data()
    results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
                         data[5], data[6], data[7], data[8], preprocess=False)
    assert len(results) == 2
