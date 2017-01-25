from collections import namedtuple

# TODO: establish standardize object for handling models used for general
# regression purposes. This will truly make the code much more modular.

# Because scipy models don't hold information on residuals or rmse, we should
# carry them forward with the models themselves, so we don't have to
# recalculate them all the time
# TODO: give better names to avoid model.model.predict nonsense
FittedModel = namedtuple('FittedModel', ['fitted_model', 'residual', 'rmse'])

# Structure to store the results, this works better than a simple
# list or tuple as it clearly states what is what
# {start_day: int,
#               end_day: int,
#               break_day: int,
#               observation_count: int,
#               change_probability: float,
#               num_coefficients: int,
#               blue:     {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               green:    {magnitude: float,
#                          rmse: float,
#                          coefficients: (float, float, ...),
#                          intercept: float},
#               red:      {magnitude: float,
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

SpectralModel = namedtuple('SpectralModel', ['rmse',
                                             'coefficients',
                                             'intercept'])

ChangeModel = namedtuple('ChangeModel', ['start_day',
                                         'end_day',
                                         'break_day',
                                         'observation_count',
                                         'change_probability',
                                         'num_coefficients',
                                         'blue',
                                         'green',
                                         'red',
                                         'nir',
                                         'swir1',
                                         'swir2',
                                         'thermal',
                                         'median_resids'])


def results_to_changemodel(fitted_models, start_day, end_day, break_day, magnitudes,
                           observation_count, change_probability, num_coefficients):
    """
    Helper method to consolidate results into a concise, self documenting data structure

    Args:
        fitted_models:
        start_day:
        end_day:
        break_day:
        observation_count:
        change_probability:
        num_coefficients:

    Returns:

    """
    spectral_models = []
    for ix, model in enumerate(fitted_models):
        spectral = SpectralModel(rmse=model.rmse,
                                 coefficients=model.fitted_model.coef_,
                                 intercept=model.fitted_model.intercept_)
        spectral_models.append(spectral)

    return ChangeModel(start_day=start_day,
                       end_day=end_day,
                       break_day=break_day,
                       observation_count=observation_count,
                       change_probability=change_probability,
                       num_coefficients=num_coefficients,
                       blue=spectral_models[0],
                       green=spectral_models[1],
                       red=spectral_models[2],
                       nir=spectral_models[3],
                       swir1=spectral_models[4],
                       swir2=spectral_models[5],
                       thermal=spectral_models[6],
                       median_resids=magnitudes)

