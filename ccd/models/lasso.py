from sklearn import linear_model
import numpy as np
from cachetools import cached
from cachetools import LRUCache

cache = LRUCache(maxsize=1000)


def __coefficient_cache_key(observation_dates):
    return tuple(observation_dates)


@cached(cache=cache, key=__coefficient_cache_key)
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
    matrix[:, 0] = [np.sin(2*np.pi*t/365.25) for t in observation_dates]
    matrix[:, 1] = [np.cos(2*np.pi*t/365.25) for t in observation_dates]
    matrix[:, 2] = [t for t in observation_dates]
    return matrix


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
    # pmodel = partial_model(observation_dates)
    lasso = linear_model.Lasso(alpha=0.1)
    return lasso.fit(coefficient_matrix(observation_dates), observations)
