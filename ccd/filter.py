"""Filters for pre-processing change model inputs.

This module currently uses explicit values from the Landsat CFMask:

    - 0: clear
    - 1: water
    - 2: cloud_shadow
    - 3: snow
    - 4: cloud
    - 255: fill
"""

from ccd import app


def count_clear_or_water(quality):
    """Count clear or water data.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of clear or water observation implied by QA data.
    """
    return quality[(quality == app.QA_CLEAR) | (quality == app.QA_WATER)].shape[0]


def count_fill(quality):
    """Count fill data.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of filled observation implied by QA data.
    """
    return quality[quality == app.QA_FILL].shape[0]


def count_snow(quality):
    """Count snow data.

    Useful for determining ratio of snow:clear pixels.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of snow pixels implied by QA data
    """
    return quality[quality == app.QA_SNOW].shape[0]


def count_total(quality):
    """Count non-fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of non-fill pixels implied by QA data.
    """
    return quality[quality != app.QA_FILL].shape[0]


def ratio_clear(quality):
    """Calculate ratio of clear to non-clear pixels; exclude, fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of non-fill pixels implied by QA data.
    """
    clear_count = count_clear_or_water(quality)
    total_count = count_total(quality)
    return clear_count / total_count


def ratio_snow(quality):
    """Calculate ratio of snow to clear pixels; exclude fill and non-clear data.

    Useful for determining ratio of snow:clear pixels.

    Arguments:
        quality: CFMask quality band values.

    Returns:
        float: Value between zero and one indicating amount of
            snow-observations.
    """
    snowy_count = count_snow(quality)
    clear_count = count_clear_or_water(quality)
    return snowy_count / (clear_count + snowy_count + 0.01)


def enough_clear(quality, threshold=app.CLEAR_PCT_THREHOLD):
    """Determine if clear observations exceed threshold.

    Useful when selecting mathematical model for detection. More clear
    observations allow for models with more coefficients.

    Arguments:
        quality: quality band values.
        threshold: minimum ratio of clear/water to not-clear/water values.

    Returns:
        boolean: True if >= threshold
    """
    return ratio_clear(quality) >= threshold


def enough_snow(quality, threshold=app.SNOW_PCT_THRESHOLD):
    """Determine if snow observations exceed threshold.

    Useful when selecting detection algorithm.

    Arguments:
        quality: quality band values.
        threshold: minimum ratio of snow to clear/water values.

    Returns:
        boolean: True if >= threshold
    """
    return ratio_snow(quality) >= threshold


def unsaturated_index(observations):
    """bool index for unsaturated obserervations between 0..10,000

    Useful for efficiently filtering noisy-data from arrays.

    Arguments:
        observations: time/spectra/qa major nd-array, assumed to be shaped as
            (9,n-moments) of unscaled data.

    """
    unsaturated = ((0 < observations[1, :]) & (observations[1, :] < 10000) &
                   (0 < observations[2, :]) & (observations[2, :] < 10000) &
                   (0 < observations[3, :]) & (observations[3, :] < 10000) &
                   (0 < observations[4, :]) & (observations[4, :] < 10000) &
                   (0 < observations[5, :]) & (observations[5, :] < 10000) &
                   (0 < observations[6, :]) & (observations[6, :] < 10000))
    return unsaturated


def temperature_index(observations, min_kelvin=179.95, max_kelvin=343.85):
    """Provide an index of observations within a brightness temperature range.

    Thermal min/max must be provided as an unscaled value in Kelvin, the same
    units as observed data.

    The range in degrees celsius is [-93.2C,70.7C]

    Arguments:
        observations: time/spectra/qa major nd-array, assumed to be shaped as
            (9,n-moments) of unscaled data.
        min_kelvin: minimum temperature in degrees kelvin, by default 179.95K,
            -93.2C.
        max_kelvin: maximum temperature in degrees kelvin, by default 343.85K,
            70.7C.
    """
    # threshold parameters are unscaled, observations are scaled so the former
    # needs to be scaled...
    min_kelvin *= 10
    max_kelvin *= 10
    return ((min_kelvin <= observations[7, :]) &
            (observations[7, :] <= max_kelvin))


def preprocess(matrix):
    """Filter matrix for clear pixels within temp/saturation range."""
    criteria = (clear_index(matrix)
                & temperature_index(matrix)
                & unsaturated_index(matrix))
    return matrix[:, criteria]
