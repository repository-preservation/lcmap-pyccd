"""Functions and classes used by the change detection procedures

This includes things such as a calculating RMSE, bounds checking,
and other methods that are used outside of the main detection loop

This allows for a layer of abstraction, where the individual loop procedures can
be looked at from a higher level
"""

import numpy as np

from ccd import app
from ccd.models import tmask
from ccd.math_utils import euclidean_norm, euclidean_norm_sq

log = app.logging.getLogger(__name__)
defaults = app.defaults


def stable(observations, models, dates,
           variogram, t_cg=defaults.CHANGE_THRESHOLD):
    """Determine if we have a stable model to start building with

    Args:
        observations: spectral observations
        models: current representative models
        dates: date values that is covered by the models
        variogram: median variogram values
        t_cg: change threshold

    Returns: Boolean on whether stable or not
    """
    rmse = [max(variogram, model.rmse)
            for model, adj_rmse
            in zip(models, variogram)]

    check_vals = []
    for spectra, spectra_model, b_rmse in zip(observations, models, rmse):
        slope = spectra_model.model.coef_[1] * (dates[-1] - dates[0])

        check_val = (abs(slope) + abs(spectra_model.residual[0]) +
                     abs(spectra_model.residual[-1])) / b_rmse

        check_vals.append(check_val)

    return euclidean_norm(check_vals) < t_cg


def change_magnitudes(dates, observations, models,
                      variogram, detect_bands=defaults.DETECTION_BANDS):
    """
    Calculate the magnitude of change of a single point in time across
    all the spectra

    Args:
        observations: spectral observations
        models: named tuple with the scipy model, rmse, and residuals
        dates: ordinal dates associated with the observations
        variogram: median variogram values across the spectral bands
        detect_bands: spectral band index values that are used
            for detecting change

    Returns:
        1-d ndarray of values representing change magnitudes across all bands
    """

    rmse = [max(variogram, model.rmse)
            for model, adj_rmse
            in zip(models[detect_bands], variogram[detect_bands])]

    # TODO Redo and move into math_utils where appropriate
    magnitudes = np.array(shape=(len(detect_bands, )))
    for idx in detect_bands:
        mag = (observations[idx] - models[idx].model.predict(dates))
        mag /= rmse[idx]
        magnitudes[idx] = mag

    return euclidean_norm_sq(magnitudes)


def detect_change(magnitudes, change_threshold=defaults.CHANGE_THRESHOLD):
    """
    Convenience function to check if the minimum magnitude surpasses the
    threshold required to determine if it is change

    Args:
        magnitudes: magnitude values across the spectral bands
        change_threshold: threshold value to determine if change has occurred

    Returns:
        bool: True if change has been detected, else False
    """
    return np.min(magnitudes) > change_threshold


def detect_outlier(magnitudes, outlier_threshold=defaults.OUTLIER_THRESHOLD):
    """
    Convenience function to check if any of the magnitudes surpass the
    threshold to mark this date as being an outlier

    This is used to mask out values from current or future processing

    Args:
        magnitudes: 1-d ndarray of magnitude values across the spectral bands
        outlier_threshold: threshold value

    Returns:
        bool: True if these spectral values should be omitted
    """
    return any(magnitudes > outlier_threshold)


# def change_magnitudes(models, coefficient_matrix, observations):
#     """Calculate change magnitudes for each model and spectra.
#
#     Magnitude is the 2-norm of the difference between predicted
#     and observed values.
#
#     Args:
#         models: fitted models, used to predict values.
#         coefficients: pre-calculated model coefficient matrix.
#         observations: spectral values, list of spectra -> values
#         threshold: tolerance between detected values and
#             predicted ones.
#
#     Returns:
#         list: magnitude of change for each model.
#     """
#     magnitudes = []
#
#     for model, observed in zip(models, observations):
#         predicted = model.predict(coefficient_matrix)
#         # TODO (jmorton): VERIFY CORRECTNESS
#         # This approach matches what is done if 2-norm (largest sing. value)
#         magnitude = euclidean_norm((predicted-observed))
#         magnitudes.append(magnitude)
#     log.debug("calculate magnitudes".format(magnitudes))
#     return magnitudes


# def accurate(magnitudes, threshold=0.99):
#     """Are observed spectral values within the predicted values' threshold.
#
#     Convenience function used to improve readability of code.
#
#     Args:
#         magnitudes: list of magnitudes for spectra
#         threshold: tolerance between detected values and predicted ones
#
#     Returns:
#         bool: True if each model's predicted and observed values are
#             below the threshold, False otherwise.
#     """
#     below = [m < threshold for m in magnitudes]
#     log.debug("change magnitued within {0}? {1}".format(threshold, below))
#     return all(below)
#
#
# def end_index(meow_ix, meow_size):
#     """Find end index for minimum expected observation window.
#
#     Args:
#         meow_ix: starting index
#         meow_size: offset from start
#
#     Returns:
#         integer: index of last observation
#     """
#     return meow_ix + meow_size - 1


def find_time_index(dates, window, meow_size=defaults.MEOW_SIZE, day_delta=defaults.DAY_DELTA):
    """Find index in times at least one year from time at meow_ix.
    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        window: index into times, used to get day number for comparing
            times for
        meow_size: relative number of observations after meow_ix to
            begin searching for a time index
        day_delta: number of days required for a years worth of data,
            defined to be 365
    Returns:
        integer: array index of time at least one year from meow_ix,
            or None if it can't be found.
    """

    # If the last time is less than a year, then iterating through
    # times to find an index is futile.
    if not enough_time(dates, window, day_delta=day_delta):
        log.debug("insufficient time ({0} days) after \
                   times[{1}]:{2}".format(day_delta, window.start, dates[window.start]))
        return None

    if window.stop:
        end_ix = window.stop
    else:
        end_ix = window.start + meow_size

    # This seems pretty naive, if you can think of something more
    # performant and elegant, have at it!
    while end_ix < dates.shape[0] - meow_size:
        if (dates[end_ix]-dates[window.start]) >= day_delta:
            break
        else:
            end_ix += 1

    log.debug("sufficient time from times[{0}..{1}] \
               (day #{2} to #{3})".format(window.start, end_ix,
                                          dates[window.start], dates[end_ix]))

    return end_ix


def enough_samples(dates, window, meow_size=defaults.MEOW_SIZE):
    """Change detection requires a minimum number of samples (as specified
    by meow size).

    This function improves readability of logic that performs this check.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        window: slice object representing the indices that we want to look at
        meow_size: offset of last time from meow_ix

    Returns:
        bool: True if times contains enough samples after meow_ix,
        False otherwise.
    """
    return window.stop <= dates.shape[0]


def enough_time(dates, window, day_delta=defaults.DAY_DELTA):
    """Change detection requires a minimum amount of time (as specified by
    day_delta).

    This function, like `enough_samples` improves readability of logic
    that performs this check.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        window: slice object representing the indices that we want to look at
        day_delta: minimum difference between time at meow_ix and most
            recent observation.

    Returns:
        list: True if the represented time span is greater than day_delta
    """
    return (dates[-1] - dates[window.start]) >= day_delta


def determine_num_coefs(dates,
                        min_coef=defaults.COEFFICIENT_MIN,
                        mid_coef=defaults.COEFFICIENT_MID,
                        max_coef=defaults.COEFFICIENT_MAX,
                        time_scalar=defaults.TIME_SCALAR):
    """
    Determine the number of coefficients to use for the main fit procedure

    This is based mostly on the amount of time (in ordinal days) that is being
    going to be covered by the model

    This is referred to as df (degrees of freedom) in the model section

    Args:
        dates: 1-d array of representative ordinal dates
        min_coef: minimum number of coefficients
        mid_coef: mid number of coefficients
        max_coef: maximum number of coefficients
        time_scalar: used to scale the time span

    Returns:

    """
    span = (dates[-1] - dates[0]) / time_scalar

    if span < mid_coef:
        return min_coef
    elif span < max_coef:
        return mid_coef
    else:
        return max_coef


def calculate_variogram(observations):
    """
    Calculate the first order variogram/madogram across all bands

    Helper method to make subsequent code clearer

    Args:
        observations: spectral band values

    Returns:
        1-d ndarray representing the variogram values
    """
    # eventually should call the method defined in math_utils.py
    return np.median(np.abs(np.diff(observations)), axis=1)


def initialize(dates, observations, fitter_fn, tmask_matrix,
               model_window, meow_size, day_delta=defaults.DAY_DELTA):
    """Determine the window indices, models, and errors for observations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        tmask_matrix: predifined matrix of coefficients used for Tmask
        model_window: start index of time/observation window
        meow_size: offset from meow_ix, determines initial window size
        day_delta: minimum difference between time at meow_ix and most
            recent observation

    Returns:
        tuple: start, end, models, errors
    """

    # Guard against insufficent data...
    if not enough_samples(dates, model_window):
        log.debug("failed, insufficient clear observations")
        return model_window, None, None, None

    if not enough_time(dates, model_window, day_delta):
        log.debug("failed, insufficient time range")
        return model_window, None, None, None

    models = None

    while model_window.stop <= dates.shape[0] - meow_size:
        log.debug("initialize from {0}..{1}".format(model_window.start,
                                                    model_window.stop))

        # Finding a sufficient window of time needs to run
        # each iteration because the starting point
        # will increment if the model isn't stable, incrementing
        # the window stop in lock-step does not guarantee a 1-year+
        # time-range.
        model_window.stop = find_time_index(dates, model_window, meow_size)

        # Subset the data based on the current model window
        period = dates[model_window]
        spectra = observations[:, model_window]
        variogram = calculate_variogram(spectra)

        # Count outliers in the window, if there are too many outliers then
        # try again.
        outliers = tmask.tmask(period,
                               spectra,
                               tmask_matrix[model_window],
                               variogram)

        # Make sure we still have enough observations and enough time
        if not enough_time(period[~outliers], model_window, day_delta)\
                or not enough_samples(period[~outliers], model_window):
            log.debug("continue, not enough observations \
                       ({0}) after tmask".format(period[~outliers].shape[0]))
            model_window.stop += 1
            continue

        # Each spectra, although analyzed independently, all share
        # a common time-frame. Consequently, it doesn't make sense
        # to analyze one spectrum in it's entirety.
        models = [fitter_fn(period[~outliers], spectrum)
                  for spectrum in spectra[~outliers]]
        log.debug("update change models")

        # If a model is not stable, then it is possible that a disturbance
        # exists somewhere in the observation window. The window shifts
        # forward in time, and begins initialization again.
        if not stable(spectra, models, period, variogram):
            log.debug("unstable model, shift start time and retry")
            model_window.start += 1
            model_window.stop += 1
            continue
        else:
            log.debug("stable model, done.")
            break

    log.debug("initialize complete, start: {0}, stop: {1}".format(model_window.start,
                                                                  model_window.stop))

    return model_window, models


def extend(dates, observations, model_window, peek_size, fitter_fn, models):
    """Increase observation window until change is detected or
    we are out of observations

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        peek_size: look ahead for detecting change
        fitter_fn: function used to model observations
        models: previously generated models, used to calculate magnitude
        day_delta: minimum difference between time at meow_ix and most
            recent observation

    Returns:
        tuple: end index, models, and change magnitude.
    """
    # Step 2: EXTENSION.
    # The second step is to update a model until observations that do not
    # fit the model are found.

    log.debug("change detection started {0}..{1}".format(model_window.start,
                                                         model_window.stop))

    if model_window.stop is None:
        log.debug("failed, end_ix is None... initialize must have failed")
        return model_window, models, None

    if (model_window.stop + peek_size) > dates.shape[0]:
        log.debug("failed, end_index+peek_size {0}+{1} \
                   exceed available data ({2})".format(model_window.stop,
                                                       peek_size,
                                                       dates.shape[0]))
        return model_window, models, None

    while (model_window.stop + peek_size) <= dates.shape[0]:
        peek_window = slice(model_window.stop, model_window.stop + peek_size)

        log.debug("detecting change in \
                   times[{0}..{1}]".format(peek_window.start,
                                           peek_window.stop))

        period = dates[peek_window]
        # coefficient_slice = coefficients[peek_window]
        spectra_slice = observations[:, peek_window]

        df = determine_num_coefs(dates[model_window])

        magnitudes = change_magnitudes(models, coefficient_slice, spectra_slice)
        if accurate(magnitudes):
            log.debug("errors below threshold {0}..{1}+{2}".format(meow_ix,
                                                                   end_ix,
                                                                   peek_size))
            models = [fitter_fn(period, spectrum, df)
                      for spectrum in spectra_slice]
            log.debug("change model updated")
            model_window.stop += 1
        else:
            log.debug("errors above threshold – change detected {0}..{1}+{2}".format(meow_ix, end_ix, peek_size))
            break

    log.debug("extension complete, meow_ix: {0}, end_ix: {1}".format(meow_ix,
                                                                     end_ix))
    return end_ix, models, magnitudes


def lookback(dates, observations, model_window, peek_size, models, variogram,
             previous_break, outlier_mask):
    """
    Special case when there is a gap between the start of a time series model
    and the previous model break point, this can include values that were
    excluded during the initialization step

    Args:
        dates: list of ordinal days
        observations: spectral values across bands
        model_window: current window of values that is being considered
        peek_size: number of values to look at
        models: currently fitted models for the model_window
        previous_break: index value of the previous break point, or the start
            of the time series if there wasn't one
        outlier_mask: index values that are currently being masked out from
            processing

    Returns:
        slice: window of indices to be used
        ndarray: outlier indicies
    """
    for idx in range(model_window.start, previous_break, -1):
        if model_window.start - previous_break > peek_size:
            lb_size = model_window.start - previous_break
        else:
            lb_size = peek_size

        magnitudes = [change_magnitudes(dates[lb], observations[:, lb], models, variogram)
                      for lb in range(model_window.start - lb_size,
                                      model_window.start - 1, -1)]

        if detect_change(magnitudes):
            # change was detected, return to parent method
            break
        elif detect_outlier(magnitudes):
            # mask the outlier from consideration
            # update the outlier_mask
            pass

        model_window.start -= 1

    return model_window, outlier_mask

        # peek_window = (model_window.start - lb_size, model_window.start - 1)
        #
        # period = dates[peek_window]
        # spectra = observations[:, peek_window]
        #
        # magnitudes = change_magnitudes(period, spectra, models, variogram)
