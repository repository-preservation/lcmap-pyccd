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


def count_clear_or_water(qa):
    """Count clear or water data"""
    return qa[(qa < 2)].shape[0]


def count_fill(qa):
    """Count fill data"""
    fill = 255
    return qa[(qa == fill)].shape[0]


def count_snow(qa):
    """Count snow data"""
    snow = 4
    return qa[(qa == snow)].shape[0]


def count_total(qa):
    """Count non-fill data"""
    fill = 255
    return qa[(qa != fill)].shape[0]


def ratio_clear(qa):
    """Calculate ratio of clear to non-clear pixels; exclude, fill data."""
    # TODO (jmorton) Verify; does the ratio exclude fill data?
    clear_count = count_clear_or_water(qa)
    total_count = count_total(qa)
    return clear_count / total_count


def ratio_snow(qa):
    """Calculate ratio of snow to clear pixels; exclude fill and non-clear data."""
    # TODO (jmorton) Verify; does the ratio exclude fill?
    # TODO (jmorton) Do we need to add 0.01 to the result like the Matlab version?
    snowy_count = count_snow(qa)
    clear_count = count_clear_or_water(qa)
    return snowy_count / (clear_count+snowy_count)


def enough_clear(qa, threshold):
    """Determine if clear observations exceed threshold.

    Useful when selecting mathematical model for detection."""
    return ratio_clear(qa) >= threshold


def enough_snow(qa, threshold):
    """Determine if snow observations exceed threshold.

    Useful when selecting detection algorithm."""
    return ratio_snow(qa) >= threshold


def unsaturated_index(observations):
    """Produce bool index for observations that are unsaturated (values between 0..10,000)

    Useful for efficiently filtering nd arrays."""
    # TODO (jmorton) Is there a more concise way to provide this function
    #      without being explicit about the expected dimensionality of the
    #      observations?
    unsaturated = ((0 < xs[1,:]) & (xs[1,:] < 10000) &
                   (0 < xs[2,:]) & (xs[2,:] < 10000) &
                   (0 < xs[3,:]) & (xs[3,:] < 10000) &
                   (0 < xs[4:,]) & (xs[4,:] < 10000) &
                   (0 < xs[5:,]) & (xs[5,:] < 10000) &
                   (0 < xs[6:,]) & (xs[6,:] < 10000))
    return unsaturated


def temperature_index(observations, min_kelvin=179.95, max_kelvin=343.85):
    """Provide an index of observations within a brightness temperature range.

    Thermal min/max must be provided as an unscaled value in Kelvin, the same
    units as observed data.

    For reference, the range in degrees celsius is: [-93.2C,70.7C]
    """
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


def preprocess(matrix):
    pass
