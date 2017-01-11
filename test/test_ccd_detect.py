"""
Tests for running ccd from the top level __init__.py/detect()

Sanity checks to make sure test data sets run to completion
"""
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

#
#
# def test_validate_sample_2_detection_results():
#     """Sample 2 contains two changes and should test the full path through
#     the algorithm including preprocessing"""
#     data = read_data("test/resources/sample_2.csv")
#     results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
#                          data[5], data[6], data[7], data[8])
#     assert len(results) == 3, "expected: {}, actual: {}".format(2,
#                                                                 len(results))
#
#
# def test_validate_sample_2_algorithm_field():
#     """Tests sample 2 again for two changes and verifies the algorithm field"""
#     data = read_data("test/resources/sample_2.csv")
#     results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
#                          data[5], data[6], data[7], data[8])
#
#     from ccd import __algorithm__
#     reported = results[0]['algorithm']
#     actual = __algorithm__
#     msg = "reported algorithm {0} did not match actual {1}".format(reported,
#                                                                    actual)
#     assert reported == actual, msg
