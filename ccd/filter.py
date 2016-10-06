"""Filters for pre-processing change model inputs.

This module currently uses explicit values from the Landsat CFMask:

    - 0: clear
    - 1: water
    - 2: cloud_shadow
    - 3: snow
    - 4: cloud
    - 255: fill
"""

import numpy as np


def count_clear_or_water(quality):
    """Count clear or water data.

    Arguments:
        quality: CFMask quality band values.

    Returns:
        integer: number of clear or water observation implied by QA data.
    """
    return quality[(quality < 2)].shape[0]


def count_fill(quality):
    """Count fill data.

    Arguments:
        quality: CFMask quality band values.

    Returns:
        integer: number of filled observation implied by QA data.
    """
    fill = 255
    return quality[(quality == fill)].shape[0]


def count_snow(quality):
    """Count snow data.

    Useful for determining ratio of snow:clear pixels.

    Arguments:
        quality: CFMask quality band values.

    Returns:
        integer: number of snow pixels implied by QA data
    """
    snow = 4
    return quality[(quality == snow)].shape[0]


def count_total(quality):
    """Count non-fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: CFMask quality band values.

    Returns:
        integer: number of non-fill pixels implied by QA data.
    """
    fill = 255
    return quality[(quality != fill)].shape[0]


def ratio_clear(quality):
    """Calculate ratio of clear to non-clear pixels; exclude, fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: CFMask quality band values.

    Returns:
        integer: number of non-fill pixels implied by QA data.
    """
    # TODO (jmorton) Verify; does the ratio exclude fill data?
    clear_count = count_clear_or_water(quality)
    total_count = count_total(quality)
    return clear_count / total_count


def ratio_snow(quality):
    """Calculate ratio of snow to clear pixels; exclude fill and non-clear data.

    Useful for determining ratio of snow:clear pixels.

    Arguments:
        quality: CFMask quality band values.

    Returns:
        float: Value between zero and one indicating amount of snow-observations.
    """
    # TODO (jmorton) Verify; does the ratio exclude fill?
    # TODO (jmorton) Do we need to add 0.01 to the result like the Matlab version?
    snowy_count = count_snow(quality)
    clear_count = count_clear_or_water(quality)
    return snowy_count / (clear_count+snowy_count)


def enough_clear(quality, threshold = 0.25):
    """Determine if clear observations exceed threshold.

    Useful when selecting mathematical model for detection. More clear observations
    allow for models with more coefficients.

    Arguments:
        quality: CFMask quality band values.
        threshold: minimum ratio of clear/water to not-clear/water values.

    Returns:
        float: Value between zero and one indicating amount of snow-observations.
    """
    return ratio_clear(quality) >= threshold


def enough_snow(quality, threshold = 0.75):
    """Determine if snow observations exceed threshold.

    Arguments:
        quality: CFMask quality band values.
        threshold: minimum ratio of snow to clear/water values.

    Useful when selecting detection algorithm."""
    return ratio_snow(quality) >= threshold


def clear_index(observations):
    return (observations[8,:] < 2)


def unsaturated_index(observations):
    """Produce bool index for observations that are unsaturated (values between 0..10,000)

    Useful for efficiently filtering noisy-data from arrays.

    Arguments:
        observations: time/spectra/qa major nd-array, assumed to be shaped as
            (9,n-moments) of unscaled data.

    """
    # TODO (jmorton) Is there a more concise way to provide this function
    #      without being explicit about the expected dimensionality of the
    #      observations?
    unsaturated = ((0 < observations[1,:]) & (observations[1,:] < 10000) &
                   (0 < observations[2,:]) & (observations[2,:] < 10000) &
                   (0 < observations[3,:]) & (observations[3,:] < 10000) &
                   (0 < observations[4,:]) & (observations[4,:] < 10000) &
                   (0 < observations[5,:]) & (observations[5,:] < 10000) &
                   (0 < observations[6,:]) & (observations[6,:] < 10000))
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
    return ((min_kelvin <= observations[7,:])&
            (observations[7,:] <= max_kelvin))


def categorize(qa):
    """ determine the category to use for detecting change """
    """
    IF clear_pct IS LESS THAN CLEAR_OBSERVATION_THRESHOLD
    THEN

        IF permanent_snow_pct IS GREATER THAN PERMANENT_SNOW_THRESHOLD
        THEN
            IF ENOUGH snow pixels
            THEN
                DO snow based change detection
            ELSE
                BAIL

    ELSE
        DO NORMAL CHANGE DETECTION
    """
    pass


def preprocess(matrix):
    """Filter matrix for clear pixels within temperature and saturation thresholds."""
    criteria = (clear_index(matrix)
                & temperature_index(matrix)
                & unsaturated_index(matrix))
    return matrix[:, criteria]
    # return t_mask(matrix[:, criteria])
