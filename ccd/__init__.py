from collections import namedtuple

# magnitude, rmse, coefficients, intercept

# all vals are tuple of floats except intercept (float)
ccd_result_band = namedtuple("CcdResultBand", ['magnitude', 'rmse',
                                               'coefficients', 'intercept'])

ccd_result = namedtuple("CcdResult", ['start_date', 'end_date',
                                      'red', 'green', 'blue',
                                      'nir', 'swir1', 'swir2',
                                      'thermal',
                                      'category'])


"""
 detections = namedtuple("Detections", ['is_change', 'is_outlier',
                                       'rmse', 'magnitude',
                                       'is_curve_start',
                                       'is_curve_end',
                                       'coefficients',
                                       'category'])

observation = namedtuple('Observation', ['coastal_aerosol', 'red', 'green',
                                         'blue', 'nir', 'swir1',
                                         'swir2', 'panchromatic',
                                         'is_cloud', 'is_clear', 'is_snow',
                                         'is_fill', 'is_water',
                                         'qa_confidence'])
"""

# all vals are tuple of floats except intercept (float)
"""
ccd_result_band = namedtuple("CcdResultBand", ['magnitude', 'rmse',
                                               'coefficients', 'intercept'])

ccd_result = namedtuple("CcdResult", ['start_date', 'end_date',
                                      'red', 'green', 'blue',
                                      'nir', 'swir1', 'swir2',
                                      'thermal',
                                      'category'])


#        results.append(models)
        result = ccd_result(start_date=start_date, end_date=end_date,
                            red=red, green=green, blue=blue,
                            nir=nir, swir1=swir1, swir2=swir2,
                            thermal=thermal, category=category)"""
