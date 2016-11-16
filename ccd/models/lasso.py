from sklearn import linear_model
import numpy as np
from cachetools import cached
from cachetools import LRUCache

cache = LRUCache(maxsize=1000)


def __coefficient_cache_key(observation_dates):
    return tuple(observation_dates)


@cached(cache=cache, key=__coefficient_cache_key)
def coefficient_matrix(observation_dates, df=4):
    """
    Args:
        observation_dates: list of ordinal dates
        df: degrees of freedom, how many coefficients to use

    Returns:
        Populated numpy array with coefficient values
    """
    w = 2 * np.pi / 365.25

    matrix = np.zeros(shape=(len(observation_dates), 8), order='F')

    matrix[:, 0] = [t for t in observation_dates]
    matrix[:, 1] = [np.cos(w*t) for t in observation_dates]
    matrix[:, 2] = [np.sin(w*t) for t in observation_dates]

    if df == 6:
        matrix[:, 3] = [np.cos(2 * w * t) for t in observation_dates]
        matrix[:, 4] = [np.sin(2 * w * t) for t in observation_dates]

    if df == 8:
        matrix[:, 5] = [np.cos(3 * w * t) for t in observation_dates]
        matrix[:, 6] = [np.sin(3 * w * t) for t in observation_dates]

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
