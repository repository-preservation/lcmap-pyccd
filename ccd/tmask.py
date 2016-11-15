import numpy as np
import sklearn.linear_model as lm
from ccd.app import logging, config

log = logging.getLogger(__name__)


def tmask_coefficient_matrix(dates):
    """Coefficient matrix that is used for Tmask modeling

    Args:
        dates: list of ordinal julian dates

    Returns:
        Populated numpy array with coefficient values
    """
    annual_cycle = 2*np.pi/365.25
    observation_cycle = annual_cycle / np.ceil((dates[-1] - dates[0]) / 365.25)

    matrix = np.zeros(shape=(len(dates), 5), order='F')
    matrix[:, 0] = [np.cos(annual_cycle*t) for t in dates]
    matrix[:, 1] = [np.sin(annual_cycle*t) for t in dates]
    matrix[:, 2] = [np.cos(observation_cycle*t) for t in dates]
    matrix[:, 3] = [np.sin(observation_cycle*t) for t in dates]

    return matrix


# TODO (jmorton) have a set of constants for array
# indexes based on what is passed in.

def tmask(dates, observations, tmask_matrix, adjusted_rmse, idx_slice, bands=(config.GREEN_IDX, config.SWIR_1_IDX)):
    """Produce an index for filtering outliers.

    Arguments:
        observations: time/spectra/qa major nd-array, assumed to be
            shaped as (9,n-moments) of unscaled data.
        bands: list of band indices used for outlier detection, by default
            bands 2 and 5.
        thresholds: list of values corresponding to bands
            used for outlier deterction.

    Return: indexed array, excluding outlier observations.
    """
    # Time and expected values using a four-part matrix of coefficients.
    regression = lm.LinearRegression()

    relevent_obs = observations[idx_slice, :]
    relevent_matrix = tmask_matrix[idx_slice, :]

    # Accumulator for outliers. This starts off as a list of False values
    # because we don't assume anything is an outlier.
    _, sample_count = observations.shape
    outliers = np.zeros(sample_count, dtype=bool)

    # For each band, determine if the delta between predeicted and actual
    # values exceeds the threshold. If it does, then it is an outlier.
    for band_ix, armse in zip(bands, adjusted_rmse):
        fit = regression.fit(relevent_matrix, relevent_obs[:, band_ix])
        predicted = fit.predict(relevent_matrix)
        outliers += np.abs(predicted-relevent_obs) > armse

    # Keep all observations that aren't outliers.
    return np.array(dates)[~outliers], observations[:, ~outliers]
