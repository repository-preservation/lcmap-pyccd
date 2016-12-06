import numpy as np
import sklearn.linear_model as lm
from ccd.app import logging, defaults

log = logging.getLogger(__name__)


def tmask_coefficient_matrix(dates, avg_days_yr=defaults.AVG_DAYS_YR):
    """Coefficient matrix that is used for Tmask modeling

    Args:
        dates: list of ordinal julian dates

    Returns:
        Populated numpy array with coefficient values
    """
    annual_cycle = 2*np.pi/avg_days_yr
    observation_cycle = annual_cycle / np.ceil((dates[-1] - dates[0]) / avg_days_yr)

    matrix = np.zeros(shape=(len(dates), 4), order='F')
    matrix[:, 0] = [np.cos(annual_cycle*t) for t in dates]
    matrix[:, 1] = [np.sin(annual_cycle*t) for t in dates]
    matrix[:, 2] = [np.cos(observation_cycle*t) for t in dates]
    matrix[:, 3] = [np.sin(observation_cycle*t) for t in dates]

    return matrix


def tmask(dates, observations, tmask_matrix, adjusted_rmse,
          bands=(defaults.GREEN_IDX, defaults.SWIR1_IDX)):
    """Produce an index for filtering outliers.

    Arguments:
        dates: ordinal date values associated to each n-moment in the observations
        observations: spectral values, assumed to be shaped as (n-bands,n-moments)
        tmask_matrix: input matrix for the linear regression
        bands: list of band indices used for outlier detection, by default
            bands 2 and 5.
        adjusted_rmse: list of values corresponding to bands
            used for outlier detection.

    Return: indexed array, excluding outlier observations.
    """
    # Time and expected values using a four-part matrix of coefficients.
    regression = lm.LinearRegression()

    # Accumulator for outliers. This starts off as a list of False values
    # because we don't assume anything is an outlier.
    _, sample_count = observations.shape
    outliers = np.zeros(sample_count, dtype=bool)

    # For each band, determine if the delta between predeicted and actual
    # values exceeds the threshold. If it does, then it is an outlier.
    for band_ix in bands:
        fit = regression.fit(tmask_matrix, observations[band_ix])
        predicted = fit.predict(tmask_matrix)
        outliers += np.abs(predicted - observations[band_ix]) > adjusted_rmse[band_ix]

    # Keep all observations that aren't outliers.
    return outliers
    # return dates[~outliers], observations[:, ~outliers]
