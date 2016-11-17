from collections import namedtuple

# Since scipy models don't hold information on residuals or rmse, we need to
# carry them forward with the models themselves, so we don't have to recalculate
# them all the time
Model = namedtuple('Model', ['model', 'residual', 'rmse'])
