from collections import namedtuple

# Since scipy models don't hold information on residuals or rmse, we should
# carry them forward with the models themselves, so we don't have to
# recalculate them all the time
# TODO: give better names to avoid model.model.fit nonsense
FittedModel = namedtuple('FittedModel', ['model', 'residual', 'rmse'])

# Structure to store the results, this works better than a simple
# list or tuple as it clearly states what is what
ChangeModel = namedtuple('ChangeModel', ['start', 'end', 'break', 'coefficients',
                                         'rmse', 'magnitude', 'probability',
                                         'num_observations'])
