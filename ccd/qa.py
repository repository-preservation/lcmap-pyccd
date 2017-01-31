"""Filters for pre-processing change model inputs.

# TODO finish fixing this
This module currently uses explicit values from the Landsat CFMask:

    - 0: clear
    - 1: water
    - 2: cloud_shadow
    - 3: snow
    - 4: cloud
    - 255: fill
"""
import numpy as np

from ccd.app import defaults
from ccd.math_utils import calc_median


def mask_snow(quality, snow=defaults.QA_SNOW):
    """
    Filter all indices that are not snow

    Args:
        quality: 1-d ndarray of values representing the quality of the
            associated spectral observations
        snow: int value that denotes snow

    Returns:
        1-d boolean ndarray showing which values are snow
    """
    return quality == snow


def mask_clear(quality, clear=defaults.QA_CLEAR):
    """
    Filter all indices that are not clear

    Args:
        quality: 1-d ndarray of values representing the quality of the
            associated spectral observations
        clear: int value that denotes clear

    Returns:
        1-d boolean ndarray showing which values are clear
    """
    return quality == clear


def mask_water(quality, water=defaults.QA_WATER):
    """
    Filter all indices that are not water

    Args:
        quality: 1-d ndarray of values representing the quality of the
            associated spectral observations
        water: int value that denotes water

    Returns:
        1-d boolean ndarray showing which values are water
    """
    return quality == water


def mask_fill(quality, fill=defaults.QA_FILL):
    """
    Filter all indices that are not fill

    Args:
        quality: 1-d ndarray of values representing the quality of the
            associated spectral observations
        fill: int value that denotes fill

    Returns:
        1-d boolean ndarray showing which values are fill
    """
    return quality == fill


def mask_clear_or_water(quality):
    """
    Filter all indices that are not fill

    Args:
        quality: 1-d ndarray of values representing the quality of the
            associated spectral observations
        fill: int value that denotes fill

    Returns:
        1-d boolean ndarray showing which values are fill
    """
    return mask_clear(quality) | mask_water(quality)


def count_clear_or_water(quality):
    """Count clear or water data.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of clear or water observation implied by QA data.
    """
    return np.sum([mask_clear(quality), mask_water(quality)])


def count_fill(quality):
    """Count fill data.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of filled observation implied by QA data.
    """
    return np.sum(mask_fill(quality))


def count_snow(quality):
    """Count snow data.

    Useful for determining ratio of snow:clear pixels.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of snow pixels implied by QA data
    """
    return np.sum(mask_snow(quality))


def count_total(quality):
    """Count non-fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of non-fill pixels implied by QA data.
    """
    return np.sum(~mask_fill(quality))


def ratio_clear(quality):
    """Calculate ratio of clear to non-clear pixels; exclude, fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: quality band values.

    Returns:
        integer: number of non-fill pixels implied by QA data.
    """
    return count_clear_or_water(quality) / count_total(quality)


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
    return count_snow(quality) / (clear_count + snowy_count + 0.01)


def enough_clear(quality, threshold=defaults.CLEAR_PCT_THREHOLD):
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


def enough_snow(quality, threshold=defaults.SNOW_PCT_THRESHOLD):
    """Determine if snow observations exceed threshold.

    Useful when selecting detection algorithm.

    Arguments:
        quality: quality band values.
        threshold: minimum ratio of snow to clear/water values.

    Returns:
        boolean: True if >= threshold
    """
    return ratio_snow(quality) >= threshold


def filter_median_green(green, filter_range=defaults.MEDIAN_GREEN_FILTER):
    """
    Filter values based on the median value + some range

    Args:
        green: array of green values
        filter_range: value added to the median value, this new result is
                      used as the value for filtering

    Returns:
        1-d boolean ndarray
    """
    median = calc_median(green) + filter_range

    return green < median


def filter_saturated(observations):
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
                   (0 < observations[0, :]) & (observations[0, :] < 10000))
    return unsaturated


def filter_thermal_celsius(thermal, min_celsius=-9320, max_celsius=7070):
    """Provide an index of observations within a brightness temperature range.

    Thermal min/max must be provided as a scaled value in degrees celsius.

    The range in unscaled degrees celsius is (-93.2C,70.7C)
    The range in scaled degrees celsius is (-9320, 7070)

    Arguments:
        thermal: 1-d array of thermal values
        min_celsius: minimum temperature in degrees celsius
        max_celsius: maximum temperature in degrees celsius
    """
    return ((thermal > min_celsius) &
            (thermal < max_celsius))


def standard_procedure_filter(observations, quality, thermal_idx=defaults.THERMAL_IDX):
    """
    Filter for the initial stages of the standard procedure.

    Clear or Water
    and Unsaturated

    Temperatures are expected to be in celsius
    Args:
        observations: 2-d ndarray, spectral observations
        quality: 1-d ndarray quality information
        thermal_idx: int value identifying the thermal band in the observations

    Returns:
        1-d boolean ndarray

    """
    return (mask_clear_or_water(quality) &
            filter_thermal_celsius(observations[thermal_idx]) &
            filter_saturated(observations))


def snow_procedure_filter(observations, quality, thermal_idx=defaults.THERMAL_IDX):
    """
    Filter for initial stages of the snow procedure

    Clear or Water
    and Snow

    Args:
        observations: 2-d ndarray, spectral observations
        quality: 1-d ndarray quality information
        thermal_idx: int value identifying the thermal band in the observations

    Returns:
        1-d boolean ndarray
    """
    return (standard_procedure_filter(observations, quality, thermal_idx) |
            mask_snow(quality))


def insufficient_clear_filter(observations, quality,
                              green_idx=defaults.GREEN_IDX,
                              thermal_idx=defaults.THERMAL_IDX):
    """
    Filter for the initial stages of the insufficient clear procedure.

    The main difference being there is an additional exclusion of observations
    where the green value is > the median green + 400.

    Args:
        observations: 2-d ndarray, spectral observations
        quality: 1-d ndarray quality information
        green_idx: int value identifying the green band in the observations
        thermal_idx: int value identifying the thermal band in the observations

    Returns:
        1-d boolean ndarray
    """
    standard_mask = standard_procedure_filter(observations, quality, thermal_idx)
    green_mask = filter_median_green(observations[:, standard_mask][green_idx])

    standard_mask[standard_mask] &= green_mask

    return standard_mask
