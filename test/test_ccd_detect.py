""" Tests for running ccd from the top level __init__.py/detect() """
# from shared import read_data
# from shared import two_change_data
#
# import ccd
#
#
# def test_validate_sample_2_detection_results():
#     """ Sample 2 contains two changes and should test the full path through
#     the algorithm"""
#     data = read_data("test/resources/sample_2.csv")
#     results = ccd.detect(data[0], data[1], data[2], data[3], data[4],
#                          data[5], data[6], data[7], data[8])
#     assert len(results) != 2, "expected: !{}, actual: {}".format(2,
#                                                                  len(results))
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
