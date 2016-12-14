"""Functions and classes used by the change detection procedures

This includes things such as a calculating RMSE, bounds checking,
and other methods that are used outside of the main detection loop

This allows for a layer of abstraction, where the individual loop procedures can
be looked at from a higher level
"""

import numpy as np

from ccd import app
from ccd.models import tmask
from ccd.math_utils import euclidean_norm, euclidean_norm_sq, calculate_variogram

log = app.logging.getLogger(__name__)
defaults = app.defaults


def stable(observations, models, dates, t_cg=defaults.CHANGE_THRESHOLD):
    """Determine if we have a stable model to start building with

    Args:
        observations: spectral observations
        models: current representative models
        dates: date values that is covered by the models
        t_cg: change threshold

    Returns: Boolean on whether stable or not
    """
    variogram = calculate_variogram(observations)

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
                      detection_bands=defaults.DETECTION_BANDS,
                      comparison_rmse=None):
    """
    Calculate the magnitude of change of a single point in time across
    all the spectra

    Args:
        observations: spectral observations
        models: named tuple with the scipy model, rmse, and residuals
        dates: ordinal dates associated with the observations
        detection_bands: spectral band index values that are used
            for detecting change

    Returns:
        1-d ndarray of values representing change magnitudes across all bands
    """
    variogram = calculate_variogram(observations)

    if comparison_rmse:
        rmse = [max(adj_rmse, comp_rmse)
                for comp_rmse, adj_rmse
                in zip(comparison_rmse, variogram[detection_bands])]
    else:
        rmse = [max(adj_rmse, model.rmse)
                for model, adj_rmse
                in zip(models[detection_bands], variogram[detection_bands])]

    # TODO Redo and move into math_utils where appropriate
    magnitudes = np.array(shape=(len(detection_bands, )))
    for idx in detection_bands:
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
                        num_obs_factor=defaults.NUM_OBS_FACTOR):
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
        num_obs_factor: used to scale the time span

    Returns:

    """
    span = dates.shape[0] / num_obs_factor

    if span < mid_coef:
        return min_coef
    elif span < max_coef:
        return mid_coef
    else:
        return max_coef


def update_processing_mask(mask, index):
    """
    Update the persistent processing mask

    This method will create a new view object as to avoid mutability issues

    Args:
        mask: 1-d boolean ndarray, current mask being used
        indexes: int/list/tuple of index(es) to be excluded from processing

    Returns:
        1-d boolean ndarry
    """
    m = mask[:]
    m[index] = 0
    return m


def initialize(dates, observations, fitter_fn, tmask_matrix,
               model_window, meow_size, processing_mask,
               day_delta=defaults.DAY_DELTA):
    """Determine the window indices, models, and errors for observations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        fitter_fn: function used for the regression portion of the algorithm
        tmask_matrix: predifined matrix of coefficients used for Tmask
        model_window: start index of time/observation window
        meow_size: offset from meow_ix, determines initial window size
        processing_mask: 1-d boolean array identifying which values to consider for processing
        day_delta: minimum difference between time at meow_ix and most
            recent observation

    Returns:
        slice object representing the start and end of a stable time segment to start with
    """

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    # Guard against insufficent data...
    if not enough_samples(period, model_window):
        log.debug("failed, insufficient clear observations")
        return model_window

    if not enough_time(period, model_window, day_delta):
        log.debug("failed, insufficient time range")
        return model_window

    while model_window.stop <= period.shape[0] - meow_size:
        log.debug("initialize from {0}..{1}".format(model_window.start,
                                                    model_window.stop))

        # Finding a sufficient window of time needs to run
        # each iteration because the starting point
        # will increment if the model isn't stable, incrementing
        # the window stop in lock-step does not guarantee a 1-year+
        # time-range.
        model_window.stop = find_time_index(dates, model_window, meow_size)

        # Subset the data based on the current model window
        model_period = period[model_window]
        model_spectral = spectral_obs[:, model_window]
        variogram = calculate_variogram(model_spectral)

        # Count outliers in the window, if there are too many outliers then
        # try again.
        outliers = tmask.tmask(model_period,
                               model_spectral,
                               tmask_matrix[model_window],
                               variogram)

        # Make sure we still have enough observations and enough time
        if not enough_time(model_period[~outliers], model_window, day_delta)\
                or not enough_samples(model_period[~outliers], model_window):
            log.debug("continue, not enough observations \
                       ({0}) after tmask".format(model_period[~outliers].shape[0]))
            model_window.stop += 1
            continue

        # Each spectra, although analyzed independently, all share
        # a common time-frame. Consequently, it doesn't make sense
        # to analyze one spectrum in it's entirety.
        models = [fitter_fn(model_period[~outliers], spectrum)
                  for spectrum in model_spectral[~outliers]]
        log.debug("update change models")

        # If a model is not stable, then it is possible that a disturbance
        # exists somewhere in the observation window. The window shifts
        # forward in time, and begins initialization again.
        if not stable(model_spectral, models, model_period, variogram):
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


def extend(dates, observations, model_window, peek_size, fitter_fn,
           processing_mask, detection_bands=defaults.DETECTION_BANDS):
    """Increase observation window until change is detected or
    we are out of observations

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        model_window: span of indices that is represented in the current
            process
        peek_size: look ahead for detecting change
        fitter_fn: function used to model observations
        processing_mask: 1-d boolean array identifying which values to consider for processing
        detection_bands: spectral band indicies that are used in the detect change portion

    Returns:
        tuple: end index, models, and change magnitude.
    """
    # Step 2: EXTENSION.
    # The second step is to update a model until observations that do not
    # fit the model are found.

    log.debug("change detection started {0}..{1}".format(model_window.start,
                                                         model_window.stop))

    # if model_window.stop is None:
    #     log.debug("failed, end_ix is None... initialize must have failed")
    #     return model_window, None, None

    if (model_window.stop + peek_size) > dates.shape[0]:
        log.debug("failed, end_index+peek_size {0}+{1} \
                   exceed available data ({2})".format(model_window.stop,
                                                       peek_size,
                                                       dates.shape[0]))
        return model_window, None, None

    time_span = 0
    outliers = []
    fit_window = model_window
    magnitudes = None

    while (model_window.stop + peek_size) <= dates.shape[0]:
        period = dates[processing_mask]
        spectral_obs = observations[:, processing_mask]

        num_coefs = determine_num_coefs(period[model_window])
        tmpcg_rmse = []
        peek_window = slice(model_window.stop, model_window.stop + peek_size)

        # The models generated during initialization cannot be used as they
        # have values that could've been masked by Tmask. Models are
        # additionally only updated during certain time frames, in this case
        # if we have < than 24 data points
        if not time_span or model_window.stop - model_window.start < 24:
            models = [fitter_fn(period[fit_window], spectrum, num_coefs)
                      for spectrum in spectral_obs[:, fit_window]]

            time_span = period[model_window.stop] - period[model_window.start]

            magnitudes = [change_magnitudes(period[idx],
                                            spectral_obs[:, idx],
                                            models)
                          for idx in range(peek_window.start,
                                           peek_window.stop)]
        # More than 24 points
        else:
            # TODO FIX ME! Need a retrain decision method to determine if we
            # should retrain a model or not
            if period[model_window.stop] - period[model_window.start] >= 1.33 * period[fit_window.stop] - period[fit_window.start]:
                fit_window.stop = model_window.stop

                models = [fitter_fn(period[fit_window], spectrum, num_coefs)
                          for spectrum in spectral_obs[:, fit_window]]

            # We need the last 24 residual values that were generated during
            # the model building step. These are temporally the closest values
            # that will be associated with value that is under scrutiny
            # TODO Make better! and paramaterize
            for band in detection_bands:
                tmp_rmse = euclidean_norm(models[band].residual[24:])
                tmp_rmse /= 4
                tmpcg_rmse.append(tmp_rmse)

            magnitudes = [change_magnitudes(period[idx],
                                            spectral_obs[:, idx],
                                            models, comparison_rmse=tmpcg_rmse)
                          for idx in range(peek_window.start,
                                           peek_window.stop)]

        log.debug("detecting change in \
                   times[{0}..{1}]".format(peek_window.start,
                                           peek_window.stop))

        if detect_change(magnitudes):
            # change was detected, return to parent method
            break
        elif detect_outlier(magnitudes):
            # keep track of any outliers as they will be excluded from future
            # processing steps
            outliers.append(peek_window.start)
            processing_mask = update_processing_mask(processing_mask, peek_window.start)

        model_window.stop += 1

    return model_window, models, magnitudes, outliers


def lookback(dates, observations, model_window, peek_size, models,
             previous_break, processing_mask):
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
        processing_mask: index values that are currently being masked out from
            processing

    Returns:
        slice: window of indices to be used
        array: indices of data that have been flagged as outliers
    """
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    outlier_indices = []
    for idx in range(model_window.start, previous_break, -1):
        if model_window.start - previous_break > peek_size:
            lb_size = model_window.start - previous_break
        else:
            lb_size = peek_size

        magnitudes = [change_magnitudes(period[lb],
                                        spectral_obs[:, lb],
                                        models)
                      for lb in range(model_window.start - lb_size,
                                      model_window.start - 1, -1)]

        if detect_change(magnitudes):
            # change was detected, return to parent method
            break
        elif detect_outlier(magnitudes):
            # keep track of any outliers as they will be excluded from future
            # processing steps
            outlier_indices.append(idx)

        # TODO verify how an outlier should affect the starting point
        model_window.start = idx

    return model_window, outlier_indices

