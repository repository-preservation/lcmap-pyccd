import numpy as np
import sklearn.linear_model as lm
import ccd.app as app

log = app.logging.getLogger(__name__)


def robust_fit_coefficient_matrix(observation_dates):
    """c1 * sin(t/365.25) + c2 * cos(t/365.25) + c3*t + c4 * 1

    Args:
        observation_dates: list of ordinal dates

    Returns:
        Populated numpy array with coefficient values
    """

    annual_cycle = 2*np.pi/365.25
    observation_cycle = annual_cycle / len(observation_dates)

    matrix = np.ones(shape=(len(observation_dates), 5))
    matrix[:, 0] = [np.cos(annual_cycle*t) for t in observation_dates]
    matrix[:, 1] = [np.sin(annual_cycle*t) for t in observation_dates]
    matrix[:, 2] = [np.cos(observation_cycle*t) for t in observation_dates]
    matrix[:, 3] = [np.sin(observation_cycle*t) for t in observation_dates]
    matrix[:, 4] = [t for t in observation_dates]
    return matrix


# TODO (jmorton) have a set of constants for array
# indexes based on what is passed in.

def tmask(times, observations, tmask_matrix, adjusted_rmse, bands=(app.GREEN_IDX, app.SWIR_1_IDX)):
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

    # TODO (jmorton) Determine how to use year parameter from original matlab
    #                code... currently using all observations.
    # TODO (jmorton) Determine suitable defaults for thresholds, the values
    #                are completely arbitrary.

    # Time and expected values using a four-part matrix of coefficients.
    regression = lm.LinearRegression()

    # Accumulator for outliers. This starts off as a list of False values
    # because we don't assume anything is an outlier.
    _, sample_count = observations.shape
    outliers = np.zeros(sample_count, dtype=bool)

    # For each band, determine if the delta between predeicted and actual
    # values exceeds the threshold. If it does, then it is an outlier.
    for band_ix, armse in zip(bands, adjusted_rmse):
        actual = observations[band_ix, :]
        fit = regression.fit(tmask_matrix, actual)
        predicted = fit.predict(tmask_matrix)
        outliers = outliers + (abs(predicted-actual) > armse)

    # Keep all observations that aren't outliers.
    return np.array(times)[~outliers], observations[:, ~outliers]
