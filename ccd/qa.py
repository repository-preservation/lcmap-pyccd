"""Filters for pre-processing change model inputs.
"""
import logging

import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt

from ccd.math_utils import calc_median, mask_value, count_value, mask_duplicate_values


log = logging.getLogger(__name__)


def checkbit(packedint, offset):
    """
    Check for a bit flag in a given int value.
    
    Args:
        packedint: bit packed int
        offset: binary offset to check

    Returns:
        bool
    """
    bit = 1 << offset

    return (packedint & bit) > 0


def qabitval(packedint, proc_params):
    """
    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value.
    
    fill > cloud > shadow > snow > water > clear
    
    Args:
        packedint: int value to bit check
        proc_params: dictionary of processing parameters

    Returns:
        offset value to use
    """
    if checkbit(packedint, proc_params.QA_FILL):
        return proc_params.QA_FILL
    elif checkbit(packedint, proc_params.QA_CLOUD):
        return proc_params.QA_CLOUD
    elif checkbit(packedint, proc_params.QA_SHADOW):
        return proc_params.QA_SHADOW
    elif checkbit(packedint, proc_params.QA_SNOW):
        return proc_params.QA_SNOW
    elif checkbit(packedint, proc_params.QA_WATER):
        return proc_params.QA_WATER
    elif checkbit(packedint, proc_params.QA_CLEAR):
        return proc_params.QA_CLEAR
    # L8 Cirrus and Terrain Occlusion
    elif (checkbit(packedint, proc_params.QA_CIRRUS1) &
          checkbit(packedint, proc_params.QA_CIRRUS2)):
        return proc_params.QA_CLEAR
    elif checkbit(packedint, proc_params.QA_OCCLUSION):
        return proc_params.QA_CLEAR

    else:
        raise ValueError('Unsupported bitpacked QA value {}'.format(packedint))


def unpackqa(quality, proc_params):
    """
    Transform the bit-packed QA values into their bit offset.
    
    Args:
        quality: 1-d array or list of bit-packed QA values
        proc_params: dictionary of processing parameters

    Returns:
        1-d ndarray
    """

    return np.array([qabitval(q, proc_params) for q in quality])


def count_clear_or_water(quality, clear, water):
    """
    Count clear or water data.

    Arguments:
        quality: quality band values.
        clear: value that represents clear
        water: value that represents water

    Returns:
        int
    """
    return count_value(quality, clear) + count_value(quality, water)


def count_total(quality, fill):
    """
    Count non-fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: quality band values.
        fill: value that represents fill

    Returns:
        int
    """
    return np.sum(~mask_value(quality, fill))


def ratio_clear(quality, clear, water, fill):
    """
    Calculate ratio of clear to non-clear pixels; exclude, fill data.

    Useful for determining ratio of clear:total pixels.

    Arguments:
        quality: quality band values.
        clear: value that represents clear
        water: value that represents water
        fill: value that represents fill

    Returns:
        int
    """
    return (count_clear_or_water(quality, clear, water) /
            count_total(quality, fill))


def ratio_snow(quality, clear, water, snow):
    """Calculate ratio of snow to clear pixels; exclude fill and non-clear data.

    Useful for determining ratio of snow:clear pixels.

    Arguments:
        quality: CFMask quality band values.
        clear: value that represents clear
        water: value that represents water
        snow: value that represents snow

    Returns:
        float: Value between zero and one indicating amount of
            snow-observations.
    """
    snowy_count = count_value(quality, snow)
    clear_count = count_clear_or_water(quality, clear, water)

    return snowy_count / (clear_count + snowy_count + 0.01)


def ratio_cloud(quality, fill, cloud):
    """
    Calculate the ratio of observations that are cloud.

    Args:
        quality: 1-d ndarray of quality information, cannot be bitpacked
        fill: int value representing fill
        cloud: int value representing cloud

    Returns:
        float
    """
    cloud_count = count_value(quality, cloud)
    total = count_total(quality, fill)

    return cloud_count / total


def ratio_water(quality, clear, water):
    """
        Calculate the ratio of observations that are water.

        Args:
            quality: 1-d ndarray of quality information, cannot be bitpacked
            clear: int value representing clear
            water: int value representing water

        Returns:
            float
        """
    clear_count = count_clear_or_water(quality, clear, water)
    water_count = count_value(quality, water)

    return water_count / (clear_count + 0.01)


def enough_clear(quality, clear, water, fill, threshold):
    """
    Determine if clear observations exceed threshold.

    Useful when selecting mathematical model for detection. More clear
    observations allow for models with more coefficients.

    Arguments:
        quality: quality band values.
        clear: value that represents clear
        water: value that represents water
        fill: value that represents fill
        threshold: minimum ratio of clear/water to not-clear/water values.

    Returns:
        boolean: True if >= threshold
    """
    return ratio_clear(quality, clear, water, fill) >= threshold


def enough_snow(quality, clear, water, snow, threshold):
    """
    Determine if snow observations exceed threshold.

    Useful when selecting detection algorithm.

    Arguments:
        quality: quality band values.
        clear: value that represents clear
        water: value that represents water
        snow: value that represents snow
        threshold: minimum ratio of snow to clear/water values.

    Returns:
        boolean: True if >= threshold
    """
    return ratio_snow(quality, clear, water, snow) >= threshold


def filter_median_green(green, filter_range):
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
    """
    bool index for unsaturated obserervations between 0..10,000

    Useful for efficiently filtering noisy-data from arrays.

    Arguments:
        observations: spectra nd-array, assumed to be shaped as
            (6,n-moments) of unscaled data.
            
    Returns:
        1-d bool ndarray

    """
    unsaturated = ((0 < observations[1, :]) & (observations[1, :] < 10000) &
                   (0 < observations[2, :]) & (observations[2, :] < 10000) &
                   (0 < observations[3, :]) & (observations[3, :] < 10000) &
                   (0 < observations[4, :]) & (observations[4, :] < 10000) &
                   (0 < observations[5, :]) & (observations[5, :] < 10000) &
                   (0 < observations[0, :]) & (observations[0, :] < 10000))
    return unsaturated


def filter_thermal_celsius(thermal, min_celsius=-9320, max_celsius=7070):
    """
    Provide an index of observations within a brightness temperature range.

    Thermal min/max must be provided as a scaled value in degrees celsius.

    The range in unscaled degrees celsius is (-93.2C,70.7C)
    The range in scaled degrees celsius is (-9320, 7070)

    Arguments:
        thermal: 1-d array of thermal values
        min_celsius: minimum temperature in degrees celsius
        max_celsius: maximum temperature in degrees celsius
        
    Returns:
        1-d bool ndarray
    """
    return ((thermal > min_celsius) &
            (thermal < max_celsius))


def standard_procedure_filter(observations, quality, dates, proc_params):
    """
    Filter for the initial stages of the standard procedure.

    Clear or Water
    and Unsaturated

    Temperatures are expected to be in celsius
    Args:
        observations: 2-d ndarray, spectral observations
        quality: 1-d ndarray observation quality information
        dates: 1-d ndarray ordinal observation dates
        proc_params: dictionary of processing parameters

    Returns:
        1-d boolean ndarray
    """
    thermal_idx = proc_params.THERMAL_IDX
    clear = proc_params.QA_CLEAR
    water = proc_params.QA_WATER

    mask = ((mask_value(quality, water) | mask_value(quality, clear)) &
            filter_thermal_celsius(observations[thermal_idx]) &
            filter_saturated(observations))

    log.debug('Obs after QA: %s', np.sum(mask))

    date_mask = mask_duplicate_values(dates[mask])
    mask[mask] = date_mask
    log.debug('Number of duplicate dates: %s', np.sum(date_mask))

    res_mask = resamplemask(dates[mask], observations[:, mask], proc_params)
    log.debug('Obs after date resampling: %s', np.sum(res_mask))
    mask[mask] = res_mask

    return mask


def snow_procedure_filter(observations, quality, dates, proc_params):
    """
    Filter for initial stages of the snow procedure

    Clear or Water
    and Snow

    Args:
        observations: 2-d ndarray, spectral observations
        quality: 1-d ndarray quality information
        dates: 1-d ndarray ordinal observation dates
        thermal_idx: int value identifying the thermal band in the observations
        proc_params: dictionary of processing parameters

    Returns:
        1-d boolean ndarray
    """
    thermal_idx = proc_params.THERMAL_IDX
    clear = proc_params.QA_CLEAR
    water = proc_params.QA_WATER
    snow = proc_params.QA_SNOW

    mask = ((mask_value(quality, water) | mask_value(quality, clear)) &
            filter_thermal_celsius(observations[thermal_idx]) &
            filter_saturated(observations)) | mask_value(quality, snow)

    date_mask = mask_duplicate_values(dates[mask])

    mask[mask] = date_mask

    return mask


def insufficient_clear_filter(observations, quality, dates, proc_params):
    """
    Filter for the initial stages of the insufficient clear procedure.

    The main difference being there is an additional exclusion of observations
    where the green value is > the median green + 400.

    Args:
        observations: 2-d ndarray, spectral observations
        quality: 1-d ndarray quality information
        dates: 1-d ndarray ordinal observation dates
        proc_params: dictionary of processing parameters

    Returns:
        1-d boolean ndarray
    """
    green_idx = proc_params.GREEN_IDX
    filter_range = proc_params.MEDIAN_GREEN_FILTER

    standard_mask = standard_procedure_filter(observations, quality, dates, proc_params)
    green_mask = filter_median_green(observations[:, standard_mask][green_idx], filter_range)

    standard_mask[standard_mask] &= green_mask

    date_mask = mask_duplicate_values(dates[standard_mask])
    standard_mask[standard_mask] = date_mask

    return standard_mask


def quality_probabilities(quality, proc_params):
    """
    Provide probabilities that any given observation falls into one of three
    categories - cloud, snow, or water.

    This is mainly used in further downstream processing, and helps ensure
    consistency.

    Args:
        quality: 1-d ndarray of quality information, cannot be bitpacked
        proc_params: dictionary of global processing parameters

    Returns:
        float probability cloud
        float probability snow
        float probability water
    """
    snow = ratio_snow(quality, proc_params.QA_CLEAR, proc_params.QA_WATER,
                      proc_params.QA_SNOW)

    cloud = ratio_cloud(quality, proc_params.QA_FILL, proc_params.QA_CLOUD)

    water = ratio_water(quality, proc_params.QA_CLEAR, proc_params.QA_WATER)

    return cloud, snow, water


def resamplemask(dates, observations, proc_params):
    # Hard coded everything for right now for simple testing
    # obviously things will need to be moved to parameters file, and other
    # such work.
    nir = proc_params.NIR_IDX
    bl = proc_params.BLUE_IDX

    res_ords = resampledates(dates,
                             observations[nir],
                             observations[bl],
                             '16D',
                             'max')

    # np.isin requires 1.13, make sure to update setup.py
    return np.isin(dates, res_ords)


def resampledates(dates, nirs, blues, freq, how):
    # The criteria was just an idea (though probably a good one), would possibly
    # need to be investigated more in-depth
    df = xr.Dataset({'criteria': (['dates'], nirs / blues),
                     'ordinals': (['dates'], dates)},
                    coords={'dates': pd.to_datetime([dt.date.fromordinal(i)
                                                     for i in dates])})

    return resample_by(df, 'criteria', freq, 'dates', how)['ordinals']


_RESAMPLE_BY_METHODS = ('min', 'max', 'first', 'last', 'median',)


def resample_by(dataset, var, freq, dim, how='max',
                keep_attrs=True, **resample_kwds):
    """ Resample all variables in Dataset based on one ``var``

    For example, one could resample all Landsat bands based on the
    maximum value in the ``ndvi`` variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing variables to resample. Cannot resample any
        data containing string or object datatypes.
    var : str
        Variable name in ``dataset``
    freq : str
        Offset frequency that specifies the step-size along the resampled dimension.
        Should look like ``{N}{offset}`` where ``N`` is an (optional) integer multipler
        (default 1) and ``offset`` is any pandas date offset alias. The full list of
        offset aliases is documented in pandas [1]_.
    dim : str
        Dimension name to resample along (e.g., 'time')
    how : str or callable
        Resampling reduction method (e.g., 'mean'). The method must not alter data from
        ``dataset``, so only operations like ``first`` or ``max`` are supported.
    keep_attrs : bool, optional
        If True, the object's attributes (`attrs`) will be copied from the original
        object to the new one.
    resample_kwds
        Keyword options to pass to :py:meth:`xarray.Dataset.resample`

    Returns
    -------
    xarray.Dataset
        Resampled dataset, resampled according to ``var``

    Raises
    ------
    ValueError
        Raised when using an unsupported resampling method
    TypeError
        Raised if any dataset variables are string/bytes/object datatype

    References
    ----------
    .. [1] http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    """
    # Implementation note:
    # xarray's "argmin" and "argmax" doesn't work with more than one dimension
    # See https://github.com/pydata/xarray/issues/1388
    #
    # There should be  faster way than resampling and masking and resampling again,
    # but will need to wait on some work within `xarray`
    if how not in _RESAMPLE_BY_METHODS:
        raise ValueError("Cannot resample by '{0}'. Please choose from: {1}"
                         .format(how, ', '.join(_RESAMPLE_BY_METHODS)))

    # Without the ability to select based on argmin/argmax we need to use a
    # workaround (masking the arrays), but this doesn't work on string data
    is_str = [dv for dv in dataset.data_vars if
              dataset[dv].dtype.type in (np.str_, np.bytes_, np.object_)]
    if is_str:
        raise TypeError('Cannot use `resample_by` with string or object '
                        'datatypes (as used by {var})'
                        .format(var=', '.join(['"%s"' % dv for dv in is_str])))

    # Resample by criterion variable
    # requires python > 3.5, update setup.py
    kwds_re = {**{dim: freq}, **resample_kwds}
    var_re = getattr(dataset[[var]].resample(**kwds_re), how)()

    # Now that we know what observations have been selected by the reduction,
    # reindex the resampled data to look like the original dataset
    var_reidx = var_re.reindex_like(dataset, method='ffill')

    # Find where identical, and mask the rest so that it can be ignored when we resample
    mask = (dataset[[var]] == var_reidx[[var]])[var]
    ds_masked = dataset.where(mask)

    # Ideally we want to use 'first' instead of 'max', but it's not implemented on dask arrays
    return ds_masked.resample(**kwds_re).max()
