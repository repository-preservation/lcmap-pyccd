"""Functions and classes used by the change detection procedures

This includes things such as a calculating RMSE, bounds checking,
and other methods that are used outside of the main detection loop

This allows for a layer of abstraction, where the individual loop procedures can
be looked at from a higher level
"""

import numpy as np

from ccd import app
from ccd.models import tmask, lasso
from ccd.math_utils import euclidean_norm, euclidean_norm_sq, calculate_variogram

log = app.logging.getLogger(__name__)
defaults = app.defaults


def stable(observations, models, dates, variogram,
           t_cg=defaults.CHANGE_THRESHOLD,
           detection_bands=defaults.DETECTION_BANDS):
    """Determine if we have a stable model to start building with

    Args:
        observations: spectral observations
        models: current representative models
        dates: date values that is covered by the models
        t_cg: change threshold

    Returns: Boolean on whether stable or not
    """
    check_vals = []
    for idx in detection_bands:
        rmse_norm = max(variogram[idx], models[idx].rmse)
        slope = models[idx].fitted_model.coef_[0] * (dates[-1] - dates[0])

        check_val = (abs(slope) + abs(models[idx].residual[0]) +
                     abs(models[idx].residual[-1])) / rmse_norm

        check_vals.append(check_val)

    euc_norm = euclidean_norm_sq(check_vals)
    log.debug('Stability norm: %s, Check against: %s', euc_norm, t_cg)

    return euc_norm < t_cg


def change_magnitude(median_resids, models, variogram,
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
    if comparison_rmse:
        rmse = [max(comparison_rmse[idx], variogram[idx])
                for idx in range(variogram.shape[0])]
    else:
        rmse = [max(models[idx].rmse, variogram[idx])
                for idx in range(variogram.shape[0])]

    magnitudes = [median_resids[idx] / rmse[idx]
                  for idx in detection_bands]

    change_mag = euclidean_norm_sq(magnitudes)

    log.debug('Magnitude of change: %s', change_mag)

    return change_mag


def median_residual(dates, observations, model):
    """
    Calculate the median residual for each band

    Args:
        dates: ordinal dates associated with the observations
        observations: spectral observations
        model: named tuple with the scipy model, rmse, and residuals

    Returns:
        1-d ndarray of residuals
    """
    return np.median(observations - lasso.predict(model, dates))


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


def detect_outlier(magnitude, outlier_threshold=defaults.OUTLIER_THRESHOLD):
    """
    Convenience function to check if any of the magnitudes surpass the
    threshold to mark this date as being an outlier

    This is used to mask out values from current or future processing

    Args:
        magnitude: float, magnitude of change at a given moment in time
        outlier_threshold: threshold value

    Returns:
        bool: True if these spectral values should be omitted
    """
    return magnitude > outlier_threshold


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

        log.debug('Insufficient time ({0} days) after times[{1}]:{2}'
                  .format(day_delta, window.start, dates[window.start]))

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

    log.debug('Sufficient time from times[{0}..{1}] (day #{2} to #{3})'
              .format(window.start, end_ix, dates[window.start], dates[end_ix]))

    return end_ix


def enough_samples(dates, window, meow_size=defaults.MEOW_SIZE):
    """Change detection requires a minimum number of samples (as specified
    by meow size).

    This function improves readability of logic that performs this check.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        window: slice object representing the indices that we want to look at

    Returns:
        bool: True if times contains enough samples
        False otherwise.
    """
    return len(dates[window]) >= meow_size


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


def initialize(dates, observations, fitter_fn, model_window,
               meow_size, peek_size, processing_mask, variogram,
               day_delta=defaults.DAY_DELTA):
    """Determine the window indices, models, and errors for observations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        fitter_fn: function used for the regression portion of the algorithm
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

    log.debug('Initial %s', model_window)
    models = None
    while model_window.stop <= period.shape[0] - peek_size:
        # Finding a sufficient window of time needs to run
        # each iteration because the starting point
        # will increment if the model isn't stable, incrementing
        # the window stop in lock-step does not guarantee a 1-year+
        # time-range.
        stop = find_time_index(dates, model_window, meow_size)
        model_window = slice(model_window.start, stop)
        log.debug('Checking window: %s', model_window)

        # Subset the data based on the current model window
        model_period = period[model_window]
        model_spectral = spectral_obs[:, model_window]

        # Count outliers in the window, if there are too many outliers then
        # try again. It is important to note that the outliers caught here
        # are only temporary, only used in this initialization step.
        tmask_outliers = tmask.tmask(model_period, model_spectral, variogram)

        log.debug('Number of Tmask outliers found: %s', np.sum(tmask_outliers))

        # Make sure we still have enough observations and enough time
        if (not enough_time(model_period[~tmask_outliers], model_window, day_delta)
            or not enough_samples(model_period[~tmask_outliers], model_window)):

            log.debug('Insufficient time or observations after Tmask, '
                      'extending model window')

            model_window = slice(model_window.start, model_window.stop + 1)
            continue

        # Each spectra, although analyzed independently, all share
        # a common time-frame. Consequently, it doesn't make sense
        # to analyze one spectrum in it's entirety.
        models = [fitter_fn(model_period[~tmask_outliers], spectrum)
                  for spectrum in model_spectral[:, ~tmask_outliers]]
        log.debug('Generating models to check for stability')

        # If a model is not stable, then it is possible that a disturbance
        # exists somewhere in the observation window. The window shifts
        # forward in time, and begins initialization again.
        if not stable(model_spectral, models, model_period, variogram):
            model_window = slice(model_window.start + 1, model_window.stop + 1)
            log.debug('Unstable model, shift window to: %s', model_window)
            continue
        else:
            log.debug('Stable start found: %s', model_window)
            break

    return model_window, models


def build(dates, observations, model_window, peek_size, fitter_fn,
          processing_mask, variogram, detection_bands=defaults.DETECTION_BANDS):
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
    # Step 2: BUILD.
    # The second step is to update a model until observations that do not
    # fit the model are found.
    log.debug('Change detection started for: %s', model_window)

    # if model_window.stop is None:
    #     log.debug("failed, end_ix is None... initialize must have failed")
    #     return model_window, None, None
    # if (model_window.stop + peek_size) > dates.shape[0]:
    #     log.debug('Failed, end_index+peek_size {0}+{1} '
    #               'exceed available data ({2})'
    #               .format(model_window.stop, peek_size, dates.shape[0]))
    #     return model_window, None, None

    time_span = 0
    outliers = []
    fit_window = model_window

    models = None
    median_resids = None
    change = 0

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

            median_resids = [median_residual(period[peek_window],
                                                  spectral_obs[idx, peek_window],
                                                  models[idx])
                                  for idx in range(observations.shape[0])]

            magnitude = change_magnitude(median_resids, models, variogram)

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

            median_resids = [median_residual(period[peek_window],
                                                  spectral_obs[idx, peek_window],
                                                  models[idx])
                                  for idx in range(observations.shape[0])]

            magnitude = change_magnitude(median_resids, models, variogram,
                                         comparison_rmse=tmpcg_rmse)

        log.debug('Detecting change for %s', peek_window)

        if detect_change(magnitude):
            # change was detected, return to parent method
            change = 1
            break
        elif detect_outlier(magnitude):
            # keep track of any outliers as they will be excluded from future
            # processing steps
            outliers.append(peek_window.start)
            processing_mask = update_processing_mask(processing_mask,
                                                     peek_window.start)

        model_window.stop += 1

    return model_window, models, median_resids, change, outliers


def lookback(dates, observations, model_window, peek_size, models,
             previous_break, processing_mask, variogram):
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
    log.debug('Previous break: %s model window: %s', previous_break, model_window)

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    outlier_indices = []
    for idx in range(model_window.start - 1, previous_break - 1, -1):
        if model_window.start - previous_break > peek_size:
            peek_window = slice(model_window.start - previous_break, model_window.start)
        elif model_window.start - peek_size < 0:
            peek_window = slice(0, model_window.start)
        else:
            peek_window = slice(model_window.start - peek_size, model_window.start)

        log.debug('Considering index: %s using peek window: %s',
                  idx, peek_window)

        median_differences = [median_residual(period[peek_window],
                                              spectral_obs[idx, peek_window],
                                              models[idx])
                              for idx in range(observations.shape[0])]

        magnitude = change_magnitude(median_differences, models, variogram)

        if detect_change(magnitude):
            log.debug('Change detected for index: %s', idx)
            # change was detected, return to parent method
            break
        elif detect_outlier(magnitude):
            log.debug('Outlier detected for index: %s', idx)
            # keep track of any outliers as they will be excluded from future
            # processing steps
            outlier_indices.append(idx)
            continue

        log.debug('Including index: %s', idx)

        # TODO verify how an outlier should affect the starting point
        model_window = slice(idx, model_window.stop)

    return model_window, outlier_indices


def catch(dates, observations, peek_size, fitter_fn,
          processing_mask, variogram, start_ix):
    """
    Handle the tail end of the time series change model process.

    Args:
        results:

    Returns:

    """
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    model_window = slice(start_ix, dates.shape[0])

    # Subset the data based on the model window
    model_period = period[model_window]
    model_spectral = spectral_obs[:, model_window]

    # Count outliers in the window, if there are too many outliers then
    # try again. It is important to note that the outliers caught here
    # are only temporary, only used in this initialization step.
    outliers = tmask.tmask(model_period, model_spectral, variogram)

    models = [fitter_fn(model_period[~outliers], spectrum)
              for spectrum in model_spectral[:, ~outliers]]

    return models, np.where(outliers)


# def fit_spectral_model(dates, observations, fitter_fn,
#                        fit_window, peek_window):
#
#     models = [fitter_fn(dates[fit_window], spectrum)
#               for spectrum
#               in observations[:, fit_window]]
#
#     magnitudes_ = change_magnitudes(
#         dates[processing_mask][start_ix:model_window.start],
#         observations[processing_mask][start_ix:model_window.start],
#         models_tmp)

