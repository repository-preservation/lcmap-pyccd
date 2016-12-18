from collections import namedtuple

# Since scipy models don't hold information on residuals or rmse, we should
# carry them forward with the models themselves, so we don't have to
# recalculate them all the time
# TODO: give better names to avoid model.model.fit nonsense
FittedModel = namedtuple('FittedModel', ['model', 'residual', 'rmse'])

# Structure to store the results, this works better than a simple
# list or tuple as it clearly states what is what
# {start_day: int,
#               end_day: int,
#               break_day: int,
#               observation_count: int,
#               change_probability: float,
#               num_coefficients: int,
#               red:      {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               green:    {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               blue:     {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               nir:      {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               swir1:    {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               swir2:    {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               thermal:  {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float}}

SpectralModel = namedtuple('SpectralModel', ['magnitude',
                                             'rmse',
                                             'coefficients',
                                             'intercept'])

ChangeModel = namedtuple('ChangeModel', ['start_day',
                                         'end_day',
                                         'break_day',
                                         'observation_count',
                                         'change_probability',
                                         'num_coefficients',
                                         'red',
                                         'green',
                                         'blue',
                                         'swir1',
                                         'swir2',
                                         'thermal'])
