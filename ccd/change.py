"""Functions and classes used by the change detection procedures

This allows for a layer of abstraction, where the individual loop procedures
can be looked at from a higher level
"""

import numpy as np

from ccd import app
from ccd.models import tmask, lasso, results_to_changemodel
from ccd.math_utils import euclidean_norm, sum_of_squares

log = app.logging.getLogger(__name__)
defaults = app.defaults


def stable(models, dates, variogram,
           t_cg=defaults.CHANGE_THRESHOLD,
           detection_bands=defaults.DETECTION_BANDS):
    """Determine if we have a stable model to start building with

    Args:
        models: list of current representative/fitted models
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        dates: array of ordinal date values
        t_cg: change threshold
        detection_bands: index locations of the spectral bands that are used
            to determine stability

    Returns:
        Boolean on whether stable or not
    """
    # This could be written differently, or more performant using numpy in the
    # future
    check_vals = []
    for idx in detection_bands:
        rmse_norm = max(variogram[idx], models[idx].rmse)
        slope = models[idx].fitted_model.coef_[0] * (dates[-1] - dates[0])

        check_val = (abs(slope) + abs(models[idx].residual[0]) +
                     abs(models[idx].residual[-1])) / rmse_norm

        check_vals.append(check_val)

    euc_norm = sum_of_squares(check_vals)
    log.debug('Stability norm: %s, Check against: %s', euc_norm, t_cg)

    return euc_norm < t_cg


def change_magnitude(residuals, variogram, comparison_rmse):
    """
    Calculate the magnitude of change for multiple points in time.

    Args:
        residuals: predicted - observed values across the desired bands,
            expecting a 2-d array with each band as a row and the observations
            as columns
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        comparison_rmse: values to compare against the variogram values

    Returns:
        1-d ndarray of values representing change magnitudes
    """
    rmse = np.maximum(variogram, comparison_rmse)

    magnitudes = residuals / rmse[:, None]

    change_mag = sum_of_squares(magnitudes, axis=0)

    log.debug('Magnitudes of change: %s', change_mag)

    return change_mag


def calc_residuals(dates, observations, model):
    """
    Calculate the residuals using the fitted model.

    Args:
        dates: ordinal dates associated with the observations
        observations: spectral observations
        model: named tuple with the scipy model, rmse, and residuals

    Returns:
        1-d ndarray of residuals
    """
    # This needs to be modularized in the future.
    # Basically the model object should have a predict method with it.
    return np.abs(observations - lasso.predict(model, dates))


def detect_change(magnitudes, change_threshold=defaults.CHANGE_THRESHOLD):
    """
    Convenience function to check if the minimum magnitude surpasses the
    threshold required to determine if it is change.

    Args:
        magnitudes: change magnitude values across the observations
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
            times
        meow_size: minimum expected observation window needed to
            produce a fit.
        day_delta: number of days required for a years worth of data,
            defined to be 365
    Returns:
        integer: array index of time at least one year from meow_ix,
            or None if it can't be found.
    """

    # If the last time is less than a year, then iterating through
    # times to find an index is futile.
    if not enough_time(dates, day_delta=day_delta):
        log.debug('Insufficient time: %s', dates[-1] - dates[0])
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


def boolean_step(start, processing_mask, step):
    """
    Using the boolean mask, we need to jump either forwards or backwards
    by the step size, in True values. This means that False values are not
    counted in the step.

    Currently not used.

    Args:
        start: index value to start with in the processing mask
        processing_mask: 1-d ndarray of boolean values
        step: how many positive or True values to count, sign indicates
            direction

    Returns:
        int: index value
    """
    true_indexes = np.where(processing_mask)[0]
    loc = np.where(start == true_indexes)[0][0]
    # Do we need bounds checking here? or should that be the responsibility
    # of the calling method?

    return true_indexes[loc + step]


def enough_samples(dates, meow_size=defaults.MEOW_SIZE):
    """Change detection requires a minimum number of samples (as specified
    by meow size).

    This function improves readability of logic that performs this check.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        meow_size: minimum expected observation window needed to
            produce a fit.

    Returns:
        bool: True if times contains enough samples
        False otherwise.
    """
    return len(dates) >= meow_size


def enough_time(dates, day_delta=defaults.DAY_DELTA):
    """Change detection requires a minimum amount of time (as specified by
    day_delta).

    This function, like `enough_samples` improves readability of logic
    that performs this check.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        day_delta: minimum difference between time at meow_ix and most
            recent observation.

    Returns:
        bool: True if the represented time span is greater than day_delta
    """
    return (dates[-1] - dates[0]) >= day_delta


def determine_num_coefs(dates,
                        min_coef=defaults.COEFFICIENT_MIN,
                        mid_coef=defaults.COEFFICIENT_MID,
                        max_coef=defaults.COEFFICIENT_MAX,
                        num_obs_factor=defaults.NUM_OBS_FACTOR):
    """
    Determine the number of coefficients to use for the main fit procedure

    This is based mostly on the amount of time (in ordinal days) that is
    going to be covered by the model

    This is referred to as df (degrees of freedom) in the model section

    Args:
        dates: 1-d array of representative ordinal dates
        min_coef: minimum number of coefficients
        mid_coef: mid number of coefficients
        max_coef: maximum number of coefficients
        num_obs_factor: used to scale the time span

    Returns:
        int: number of coefficients to use during the fitting process
    """
    span = dates.shape[0] / num_obs_factor

    if span < mid_coef:
        return min_coef
    elif span < max_coef:
        return mid_coef
    else:
        return max_coef


def update_processing_mask(mask, index, window=None):
    """
    Update the persistent processing mask.

    Because processes apply the mask first, index values given are in relation
    to that. So we must apply the mask to itself, then update the boolean
    values.

    The window slice object is to catch when it is in relation to some
    window of the masked values. So, we must mask against itself, then look at
    a subset of that result.

    This method should create a new view object to avoid mutability issues.

    Args:
        mask: 1-d boolean ndarray, current mask being used
        index: int/list/tuple of index(es) to be excluded from processing,
            or boolean array
        window: slice object identifying a further subset of the mask

    Returns:
        1-d boolean ndarray
    """
    m = mask[:]
    sub = m[m]

    if window:
        sub[window][index] = False
    else:
        sub[index] = False

    m[m] = sub

    return m


def find_closest_doy(dates, date_idx, window, num):
    """
    Find the closest n dates based on day of year.

    e.g. if the date you are looking for falls on July 1, then find
    n number of dates that are closest to that same day of year.

    Args:
        dates: 1-d ndarray of ordinal day values
        date_idx: index of date value
        window: slice object identifying the subset of values used in the
            current model
        num: number of index values desired

    Returns:
        1-d ndarray of index values
    """
    # May be a better way of doing this
    d_rt = dates[window] - dates[date_idx]
    d_yr = np.abs(np.round(d_rt / 365.25) * 365.25 - d_rt)

    return np.argsort(d_yr)[:num]


def initialize(dates, observations, fitter_fn, model_window,
               meow_size, peek_size, processing_mask, variogram,
               day_delta=defaults.DAY_DELTA):
    """
    Determine a good starting point at which to build off of for the
    subsequent process of change detection, both forward and backward.

    Args:
        dates: 1-d ndarray of ordinal day values
        observations: 2-d ndarray representing the spectral values
        fitter_fn: function used for the regression portion of the algorithm
        model_window: start index of time/observation window
        meow_size: offset from meow_ix, determines initial window size
        peek_size: number of observations to consider when detecting
            a change.
        processing_mask: 1-d boolean array identifying which values to
            consider for processing
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        day_delta: minimum difference between the start and end of a model
            window

    Returns:
        slice: model window that was deemed to be a stable start
        namedtuple: fitted regression models
    """
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    log.debug('Initial %s', model_window)
    models = None
    while model_window.stop + meow_size < period.shape[0]:
        # Finding a sufficient window of time needs to run
        # each iteration because the starting point
        # will increment if the model isn't stable, incrementing only
        # the window stop in lock-step does not guarantee a 1-year+
        # time-range.
        stop = find_time_index(dates, model_window, meow_size)
        model_window = slice(model_window.start, stop)
        log.debug('Checking window: %s', model_window)

        # Count outliers in the window, if there are too many outliers then
        # try again.
        tmask_outliers = tmask.tmask(period[model_window],
                                     spectral_obs[:, model_window],
                                     variogram)

        tmask_count = np.sum(tmask_outliers)

        log.debug('Number of Tmask outliers found: %s', tmask_count)

        # Subset the data to the observations that currently under scrutiny
        # and remove the outliers identified by the tmask.
        tmask_period = period[model_window][~tmask_outliers]

        # TODO should probably look at a different fit procedure to handle
        # the following case.
        if tmask_count == model_window.stop - model_window.start:
            log.debug('Tmask identified all values as outliers')

            model_window = slice(model_window.start, model_window.stop + 1)
            continue

        # Make sure we still have enough observations and enough time after
        # the tmask removal.
        if not enough_time(tmask_period, day_delta) or not enough_samples(tmask_period):

            log.debug('Insufficient time or observations after Tmask, '
                      'extending model window')

            model_window = slice(model_window.start, model_window.stop + 1)
            continue

        # Update the persistent mask with the values identified by the Tmask
        if any(tmask_outliers):
            processing_mask = update_processing_mask(processing_mask,
                                                     tmask_outliers,
                                                     model_window)

            # The model window now actually refers to a smaller slice
            model_window = slice(model_window.start,
                                 model_window.stop - tmask_count)
            # Update the subset
            period = dates[processing_mask]
            spectral_obs = observations[:, processing_mask]

        log.debug('Generating models to check for stability')
        models = [fitter_fn(period[model_window], spectrum)
                  for spectrum in spectral_obs[:, model_window]]

        # If a model is not stable, then it is possible that a disturbance
        # exists somewhere in the observation window. The window shifts
        # forward in time, and begins initialization again.
        if not stable(models, period[model_window], variogram):
            model_window = slice(model_window.start + 1, model_window.stop + 1)
            log.debug('Unstable model, shift window to: %s', model_window)
            models = None
            continue
        else:
            log.debug('Stable start found: %s', model_window)
            break

    return model_window, models, processing_mask


def lookforward(dates, observations, model_window, peek_size, fitter_fn,
                processing_mask, variogram, detection_bands=defaults.DETECTION_BANDS):
    """Increase observation window until change is detected or
    we are out of observations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        model_window: span of indices that is represented in the current
            process
        peek_size: look ahead for detecting change
        fitter_fn: function used to model observations
        processing_mask: 1-d boolean array identifying which values to
            consider for processing
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        detection_bands: spectral band indicies that are used in the detect
            change portion

    Returns:
        namedtuple: representation of the time segment
        1-d bool ndarray: processing mask that may have been modified
        slice: model window
    """
    # Step 4: lookforward.
    # The second step is to update a model until observations that do not
    # fit the model are found.
    log.debug('lookforward initial model window: %s', model_window)

    # The fit_window pertains to which locations are used in the model
    # regression, while the model_window identifies the locations in which
    # fitted models apply to. They are not always the same.
    fit_window = model_window

    # Initialized for a check at the first iteration.
    models = None

    # Simple value to determine if change has occured or not. Change may not
    # have occurred if we reach the end of the time series.
    change = 0

    # Initial subset of the data
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    # Used for comparison purposes
    fit_span = period[model_window.stop - 1] - period[model_window.start]

    # stop is always exclusive
    while model_window.stop + peek_size < period.shape[0] or models is None:
        num_coefs = determine_num_coefs(period[model_window])

        peek_window = slice(model_window.stop, model_window.stop + peek_size)

        # Used for comparison against fit_span
        model_span = period[model_window.stop - 1] - period[model_window.start]

        log.debug('Detecting change for %s', peek_window)

        # If we have less than 24 observations covered by the model_window
        # or it the first iteration, then we always fit a new window
        if not models or model_window.stop - model_window.start < 24:
            fit_span = period[model_window.stop - 1] - period[
                model_window.start]

            fit_window = model_window
            log.debug('Retrain models, less than 24 samples')
            models = [fitter_fn(period[fit_window], spectrum, num_coefs)
                      for spectrum in spectral_obs[:, fit_window]]

            residuals = np.array([calc_residuals(period[peek_window],
                                                 spectral_obs[idx, peek_window],
                                                 models[idx])
                                  for idx in range(observations.shape[0])])

            comp_rmse = [models[idx].rmse for idx in detection_bands]

        # More than 24 points
        else:
            # If the number of observations that the current fitted models
            # expand past a threshold, then we need to fit new ones.
            # The 1.33 should be parametrized at some point.
            if model_span >= 1.33 * fit_span:
                log.debug('Retrain models, model_span: %s fit_span: %s',
                          model_span, fit_span)
                fit_span = period[model_window.stop - 1] - period[
                    model_window.start]
                fit_window = model_window

                models = [fitter_fn(period[fit_window], spectrum, num_coefs)
                          for spectrum in spectral_obs[:, fit_window]]

            residuals = np.array([calc_residuals(period[peek_window],
                                                 spectral_obs[idx, peek_window],
                                                 models[idx])
                                  for idx in range(observations.shape[0])])

            # We want to use the closest residual values to the peek_window
            # values based on seasonality.
            closest_indexes = find_closest_doy(period, peek_window.stop - 1,
                                               fit_window, 24)

            # Calculate an RMSE for the seasonal residual values, using 8
            # as the degrees of freedom.
            comp_rmse = [euclidean_norm(models[idx].residual[closest_indexes]) / 4
                         for idx in detection_bands]

        # Calculate the change magnitude values for each observation in the
        # peek_window.
        magnitude = change_magnitude(residuals[detection_bands, :],
                                     variogram[detection_bands],
                                     comp_rmse)

        if detect_change(magnitude):
            log.debug('Change detected at: %s', peek_window.start)

            # Change was detected, return to parent method
            change = 1
            break
        elif detect_outlier(magnitude[0]):
            log.debug('Outlier detected at: %s', peek_window.start)

            # Keep track of any outliers so they will be excluded from future
            # processing steps
            processing_mask = update_processing_mask(processing_mask,
                                                     peek_window.start)

            # Because only one value was excluded, we shouldn't need to adjust
            # the model_window.  The location hasn't been used in
            # processing yet. So, the next iteration can use the same windows
            # without issue.
            period = dates[processing_mask]
            spectral_obs = observations[:, processing_mask]
            continue

        model_window = slice(model_window.start, model_window.stop + 1)

    result = results_to_changemodel(fitted_models=models,
                                    start_day=period[model_window.start],
                                    end_day=period[model_window.stop - 1],
                                    break_day=period[peek_window.start],
                                    magnitudes=magnitude,
                                    observation_count=(
                                    model_window.stop - model_window.start),
                                    change_probability=change,
                                    num_coefficients=num_coefs)

    return result, processing_mask, model_window


def lookback(dates, observations, model_window, peek_size, models,
             previous_break, processing_mask, variogram,
             detection_bands=defaults.DETECTION_BANDS):
    """
    Special case when there is a gap between the start of a time series model
    and the previous model break point, this can include values that were
    excluded during the initialization step.

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
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        detection_bands: subset of spectral bands to use to detect change

    Returns:
        slice: window of indices to be used
        array: indices of data that have been flagged as outliers
    """
    log.debug('Previous break: %s model window: %s', previous_break, model_window)
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    while model_window.start > previous_break:
        # Three conditions to see how far we want to look back each iteration.
        # 1. If we have more than 6 previous observations
        # 2. Catch to make sure we don't go past the start of observations
        # 3. Less than 6 observations to look at

        # Important note about python slice objects, start is inclusive and
        # stop is exclusive, regardless of direction/step
        if model_window.start - previous_break > peek_size:
            peek_window = slice(model_window.start - 1, model_window.start - peek_size, -1)
        elif model_window.start - peek_size <= 0:
            peek_window = slice(model_window.start - 1, None, -1)
        else:
            peek_window = slice(model_window.start - 1, previous_break - 1, -1)

        log.debug('Considering index: %s using peek window: %s',
                  peek_window.start, peek_window)

        residuals = np.array([calc_residuals(period[peek_window],
                                             spectral_obs[idx, peek_window],
                                             models[idx])
                              for idx in range(observations.shape[0])])

        # log.debug('Residuals for peek window: %s', residuals)

        comp_rmse = [models[idx].rmse for idx in detection_bands]

        log.debug('RMSE values for comparison: %s', comp_rmse)

        magnitude = change_magnitude(residuals[detection_bands, :],
                                     variogram[detection_bands],
                                     comp_rmse)

        if detect_change(magnitude):
            log.debug('Change detected for index: %s', peek_window.start)
            # change was detected, return to parent method
            break
        elif detect_outlier(magnitude[0]):
            log.debug('Outlier detected for index: %s', peek_window.start)
            processing_mask = update_processing_mask(processing_mask,
                                                     peek_window.start)

            period = dates[processing_mask]
            spectral_obs = observations[:, processing_mask]

            # Because this location was used in determining the model_window
            # passed in, we must now account for removing it.
            model_window = slice(model_window.start - 1, model_window.stop - 1)
            continue

        log.debug('Including index: %s', peek_window.start)
        model_window = slice(peek_window.start, model_window.stop)

    return model_window, processing_mask


def catch(dates, observations, fitter_fn, processing_mask, model_window):
    """
    Handle special cases where general models just need to be fitted and return
    their results.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        model_window: span of indices that is represented in the current
            process
        fitter_fn: function used to model observations
        processing_mask: 1-d boolean array identifying which values to
            consider for processing

    Returns:
        namedtuple representing the time segment

    """
    log.debug('Catching observations: %s', model_window)
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    # Subset the data based on the model window
    model_period = period[model_window]
    model_spectral = spectral_obs[:, model_window]

    models = [fitter_fn(model_period, spectrum)
              for spectrum in model_spectral]

    try:
        break_day = period[model_window.stop]
    except:
        break_day = period[-1]

    result = results_to_changemodel(fitted_models=models,
                                    start_day=period[model_window.start],
                                    end_day=period[model_window.stop - 1],
                                    break_day=break_day,
                                    magnitudes=np.zeros(shape=(7,)),
                                    observation_count=(
                                        model_window.stop - model_window.start),
                                    change_probability=0,
                                    num_coefficients=4)

    return result
