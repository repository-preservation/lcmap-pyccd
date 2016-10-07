import numpy as np
import sklearn.linear_model as lm

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


# TODO (jmorton) have a set of constants for array indexes based on what is passed in.

def tmask(times, observations, adjusted_rmse, bands=(1, 4)):
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
    C = coefficient_matrix(times)
    regression = lm.LinearRegression()

    # Accumulator for outliers. This starts off as a list of False values
    # because we don't assume anything is an outlier.
    _, samples = observations.shape
    outliers = np.zeros(samples, dtype=bool)

    # For each band, determine if the delta between predeicted and actual
    # values exceeds the threshold. If it does, then it is an outlier.
    for band_ix, armse in zip(bands, adjusted_rmse):
        actual = observations[band_ix, :]
        fit = regression.fit(C, actual)
        predicted = fit.predict(C)
        outliers = outliers + (abs(predicted-actual) > armse)

    # Keep all observations that aren't outliers.
    return np.array(times)[~outliers], observations[:, ~outliers]
