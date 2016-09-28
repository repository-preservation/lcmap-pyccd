from sklearn import linear_model
import numpy as np
from functools import partial


def generate_coefficient_matrix(observation_dates):
    """ c1 * sin(t/365.25) + c2 * cos(t/365.25) + c3*t + c4 * 1

    Args:
        observation_dates: list of ordinal dates

    Returns:
        Populated numpy array with coefficient values
    """
    # c1 = np.array([np.sin(t/365.25) for t in observation_dates])
    # c2 = np.array([np.cos(t/365.25) for t in observation_dates])
    # c3 = np.array([t for t in observation_dates])
    # c4 = np.ones(len(c1))

    matrix = np.ones(shape=(len(observation_dates), 4))
    matrix[:, 0] = [np.sin(t/365.25) for t in observation_dates]
    matrix[:, 1] = [np.cos(t/365.25) for t in observation_dates]
    matrix[:, 2] = [t for t in observation_dates]
    return matrix


def initialize_model(coefficients, observations):
    """ Return a model with coefficients ready for fitting """
    return partial(linear_model.Lasso(alpha=0.1).fit(coefficients))
