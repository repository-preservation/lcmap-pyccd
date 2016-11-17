"""Functions and classes used by the change detection procedures

This includes things such as a calculating RMSE, bounds checking,
and other methods that are used outside of the main detection loop

This allows for a layer of abstraction, where the individual loop procedures can
be looked at from a higher level
"""

import numpy as np

from ccd import app
from ccd.models import tmask

log = app.logging.getLogger(__name__)
config = app.config


# def rmse(models, coefficient_matrix, observations):
#     """Calculate RMSE for all models; used to determine if models are stable.
#
#     Args:
#         models: fitted models, used to predict values, corresponds to
#             observation spectra.
#         coefficient_matrix: TODO
#         observations: list of spectra corresponding to models
#
#     Returns:
#         list: RMSE for each model.
#     """
#     errors = []
#     for model, observed in zip(models, observations):
#         predictions = model.predict(coefficient_matrix)
#         error = np.sqrt(mean_squared_error(observed, predictions))
#         errors.append(error)
#     log.debug("calculate RMSE")
#     return errors


def detect_change(observations, models, dates,
                  adjusted_rmse, t_cg=config.CHANGE_THRESHOLD):
    """Determine change has happened at a given moment in time in a
    given spectral value

    Args:
        observations: spectral observations
        models: current representative models
        dates: date values that is covered by the models
        adjusted_rmse: median variogram values
        t_cg: change threshold

    Returns: Boolean on whether change has been detected or not
    """
    rmse_thresh = [max(adjusted_rmse, model.rmse)
                   for model, adj_rmse
                   in zip(models, adjusted_rmse)]

    check_vals = []
    for spectra, spectra_model, rmse in zip(observations, models, rmse_thresh):
        slope = spectra_model.model.coef_[0] * (dates[-1] - dates[0])
        check_val = (abs(slope) + abs(spectra_model.residual[0]) +
                     abs(spectra_model.residual[-1])) / rmse
        check_vals.append(check_val)

    if np.linalg.norm(check_vals, ord=2) > t_cg:
        return True
    else:
        return False


def stable(errors, threshold=config.STABILITY_THRESHOLD):
    """Determine if all models RMSE are below threshold.

    Convenience function used to improve readability of code.

    Args:
        errors: list of error values corresponding to observation
            spectra.
        threshold: tolerance for error, all errors must be strictly
            below this value.

    Returns:
        bool: True, if all models RMSE is below threshold, False otherwise.
    """
    below = ([e < threshold for e in errors])
    log.debug("check model stability, all errors \
               below {0}? {1}".format(threshold, below))
    return all(below)


def change_magnitudes(models, coefficient_matrix, observations):
    """Calculate change magnitudes for each model and spectra.

    Magnitude is the 2-norm of the difference between predicted
    and observed values.

    Args:
        models: fitted models, used to predict values.
        coefficients: pre-calculated model coefficient matrix.
        observations: spectral values, list of spectra -> values
        threshold: tolerance between detected values and
            predicted ones.

    Returns:
        list: magnitude of change for each model.
    """
    magnitudes = []

    for model, observed in zip(models, observations):
        predicted = model.predict(coefficient_matrix)
        # TODO (jmorton): VERIFY CORRECTNESS
        # This approach matches what is done if 2-norm (largest sing. value)
        magnitude = np.linalg.norm((predicted-observed), ord=2)
        magnitudes.append(magnitude)
    log.debug("calculate magnitudes".format(magnitudes))
    return magnitudes


def accurate(magnitudes, threshold=0.99):
    """Are observed spectral values within the predicted values' threshold.

    Convenience function used to improve readability of code.

    Args:
        magnitudes: list of magnitudes for spectra
        threshold: tolerance between detected values and predicted ones

    Returns:
        bool: True if each model's predicted and observed values are
            below the threshold, False otherwise.
    """
    below = [m < threshold for m in magnitudes]
    log.debug("change magnitued within {0}? {1}".format(threshold, below))
    return all(below)


def end_index(meow_ix, meow_size):
    """Find end index for minimum expected observation window.

    Args:
        meow_ix: starting index
        meow_size: offset from start

    Returns:
        integer: index of last observation
    """
    return meow_ix + meow_size - 1


def find_time_index(dates, window, meow_size=config.MEOW_SIZE, day_delta=config.DAY_DELTA):
    """Find index in times at least one year from time at meow_ix.
    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        window: index into times, used to get day number for comparing
            times for
        meow_size: relative number of observations after meow_ix to
            begin searching for a time index
        day_delta: number of days difference between meow_ix and
            time index
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
    while end_ix < dates.shape[0]:
        if (dates[end_ix]-dates[window.start]) >= day_delta:
            break
        else:
            end_ix += 1

    log.debug("sufficient time from times[{0}..{1}] \
               (day #{2} to #{3})".format(window.start, end_ix,
                                          dates[window.start], dates[end_ix]))

    return end_ix


def enough_samples(dates, window, meow_size=config.MEOW_SIZE):
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


def enough_time(dates, window, day_delta=config.DAY_DELTA):
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


def initialize(dates, observations, fitter_fn, model_matrix, tmask_matrix,
               window, meow_size, adjusted_rmse, day_delta=config.DAY_DELTA):
    """Determine the window indices, models, and errors for observations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        model_matrix: TODO
        tmask_matrix: TODO
        window: start index of time/observation window
        meow_size: offset from meow_ix, determines initial window size
        day_delta: minimum difference between time at meow_ix and most
            recent observation

    Returns:
        tuple: start, end, models, errors
    """

    # Guard...
    if not enough_samples(dates, window):
        log.debug("failed, insufficient clear observations")
        return window, None, None, None

    if not enough_time(dates, window, day_delta):
        log.debug("failed, insufficient time range")
        return window, None, None, None

    models = None

    while window.stop <= dates.shape[0]:
        log.debug("initialize from {0}..{1}".format(window.start,
                                                    window.stop))

        # Finding a sufficient window of time needs must run
        # each iteration because the starting point
        # will increment if the model isn't stable, incrementing
        # the window of in lock-step does not guarantee a 1-year+
        # time-range.
        window.stop = find_time_index(dates, window, meow_size)

        period = dates[window]
        matrix = model_matrix[window]
        spectra = observations[:, window]

        # Count outliers in the window, if there are too many outliers then
        # try again.
        dates_, observations_ = tmask.tmask(period,
                                            spectra,
                                            tmask_matrix[window, :],
                                            adjusted_rmse)

        # Make sure we still have enough observations and enough time
        if (dates_.shape[0] < meow_size) or ((dates_[-1] - dates_[0]) < day_delta):
            log.debug("continue, not enough observations \
                       ({0}) after tmask".format(dates_.shape[0]))
            window.stop += 1
            continue

        # Each spectra, although analyzed independently, all share
        # a common time-frame. Consequently, it doesn't make sense
        # to analyze one spectrum in it's entirety.
        models = [fitter_fn(matrix, spectrum) for spectrum in spectra]
        log.debug("update change models")

        # If a model is not stable, then it is possible that a disturbance
        # exists somewhere in the observation window. The window shifts
        # forward in time, and begins initialization again.
        if detect_change(spectra, models, period, adjusted_rmse):
            log.debug("unstable model, shift start time and retry")
            window.start += 1
            window.stop += 1
            continue
        else:
            log.debug("stable model, done.")
            break

    log.debug("initialize complete, start: {0}, stop: {1}".format(window.start,
                                                                  window.stop))

    return window, models


def extend(dates, observations, coefficients,
           window, peek_size, fitter_fn, models):
    """Increase observation window until change is detected or
    we are out of observations

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        coefficients: pre-calculated model coefficients
        meow_ix: start index of time/observation window
        end_ix: end index of time/observation window
        peek_size: looked ahead for detecting change
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

    log.debug("change detection started {0}..{1}".format(window.start,
                                                         window.stop))

    if window.stop is None:
        log.debug("failed, end_ix is None... initialize must have failed")
        return window, models, None

    if (window.stop + peek_size) > dates.shape[0]:
        log.debug("failed, end_index+peek_size {0}+{1} \
                   exceed available data ({2})".format(window.stop,
                                                       peek_size,
                                                       dates.shape[0]))
        return window, models, None

    while (window.stop + peek_size) <= dates.shape[0]:
        peek_window = slice(window.start, window.stop + peek_size)

        log.debug("detecting change in \
                   times[{0}..{1}]".format(peek_window.stop,
                                           peek_window.stop))

        time_slice = dates[peek_window]
        coefficient_slice = coefficients[peek_window]
        spectra_slice = observations[:, peek_window]

        magnitudes_ = change_magnitudes(models, coefficient_slice, spectra_slice)
        if accurate(magnitudes_):
            log.debug("errors below threshold {0}..{1}+{2}".format(meow_ix,
                                                                   end_ix,
                                                                   peek_size))
            models = [fitter_fn(time_slice, spectrum) for spectrum in spectra_slice]
            log.debug("change model updated")
            window.stop += 1
        else:
            log.debug("errors above threshold – change detected {0}..{1}+{2}".format(meow_ix, end_ix, peek_size))
            break

    log.debug("extension complete, meow_ix: {0}, end_ix: {1}".format(meow_ix,
                                                                     end_ix))
    return end_ix, models, magnitudes_
