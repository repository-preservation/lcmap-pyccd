from collections import namedtuple

# TODO: establish standardize object for handling models used for general
# regression purposes. This will truly make the code much more modular.

# Because scipy models don't hold information on residuals or rmse, we should
# carry them forward with the models themselves, so we don't have to
# recalculate them all the time
# TODO: give better names to avoid model.model.predict nonsense
FittedModel = namedtuple('FittedModel', ['fitted_model', 'residual', 'rmse'])


def results_to_changemodel(fitted_models, start_day, end_day, break_day,
                           magnitudes, observation_count, change_probability,
                           curve_qa):
    """
    Helper method to consolidate results into a concise, self documenting data
    structure.

    {start_day: int,
     end_day: int,
     break_day: int,
     observation_count: int,
     change_probability: float,
     curve_qa: int,
     blue:  {magnitude: float,
             rmse: float,
             coefficients: (float, float, ...),
             intercept: float},
     etc...

    Returns:
        dict

    """
    spectral_models = []
    for ix, model in enumerate(fitted_models):
        spectral = {'rmse': model.rmse,
                    'coefficients': model.fitted_model.coef_,
                    'intercept': model.fitted_model.intercept_,
                    'magnitude': magnitudes[ix]}
        spectral_models.append(spectral)

    return {'start_day': start_day,
            'end_day': end_day,
            'break_day': break_day,
            'observation_count': observation_count,
            'change_probability': change_probability,
            'curve_qa': curve_qa,
            'blue': spectral_models[0],
            'green': spectral_models[1],
            'red': spectral_models[2],
            'nir': spectral_models[3],
            'swir1': spectral_models[4],
            'swir2': spectral_models[5],
            'thermal': spectral_models[6]}
