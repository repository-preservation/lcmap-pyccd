from sklearn import linear_model
import numpy as np
from functools import partial
from functools import lru_cache


def coefficient_matrix(observation_dates):
    """c1 * sin(t/365.25) + c2 * cos(t/365.25) + c3*t + c4 * 1

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


@lru_cache(maxsize=128, typed=True)
def partial_model(observation_dates):
    """Return a partial model with coefficients ready for fitting.

    Args:
        observation_dates: tuple (or hashable collection) of ordinal dates

    Returns:
        Partial sklearn.linear_model.Lasso().fit(observation_dates)

    Example:
        pmodel = partial_model(observation_dates)

        Then:
        fitted_model = pmodel(observations)
        fitted_model.predict(...)
    """
    lasso = linear_model.Lasso(alpha=0.1)
    return partial(lasso.fit, coefficient_matrix(observation_dates))


def fitted_model(observation_dates, observations):
    """Create a fully fitted lasso model.

    Args:
        observation_dates: list or ordinal observation dates
        observations: list of values corresponding to observation_dates

    Returns:
        sklearn.linear_model.Lasso().fit(observation_dates, observations)

    Example:
        fitted_model(dates, obs).predict(...)
    """
    pmodel = partial_model(observation_dates)
    return pmodel(observations)
