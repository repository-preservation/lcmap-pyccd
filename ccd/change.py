"""Functions for producing change model parameters.

The change module provides a 'detect' function used to produce change model
parameters for multi-spectra time-series data. It is implemented in a manner
independent of data sources, input formats, pre-processing routines, and
output formats.

In general, change detection is an iterative, two-step process: an initial
stable period of time is found for a time-series of data and then the same
window is extended until a change is detected. These steps repeat until all
available observations are considered.

The result of this process is a list-of-lists of change models that correspond
to observation spectra.

Preprocessing routines are essential to, but distinct from, the core change
detection algorithm. See the `ccd.filter` for more details related to this
step.

For more information please refer to the `CCDC Algorithm Description Document`.

.. _Algorithm Description Document:
   http://landsat.usgs.gov/documents/ccdc_add.pdf
"""

import numpy as np
import ccd.models.lasso as lasso
import ccd.tmask as tmask
from ccd import app

log = app.logging.getLogger(__name__)


def rmse(models, coefficient_matrix, observations):
    """Calculate RMSE for all models; used to determine if models are stable.

    Args:
        models: fitted models, used to predict values, corresponds to
            observation spectra.
        coefficient_matrix: TODO
        observations: list of spectra corresponding to models

    Returns:
        list: RMSE for each model.
    """
    errors = []
    for model, observed in zip(models, observations):
        predictions = model.predict(coefficient_matrix)
        # TODO (jmorton): VERIFY CORRECTNESS
        error = (np.linalg.norm(predictions - observed) /
                 np.sqrt(len(predictions)))
        errors.append(error)
    log.debug("calculate RMSE")
    return errors


def stable(errors, threshold=200.0):
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
    log.debug("check model stability, all errors below {0}? {1}".format(threshold, below))
    return all(below)


def magnitudes(models, coefficients, observations):
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
        predicted = model.predict(coefficients)
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
    log.debug("all errors below {0}? {1}".format(threshold, below))
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


def find_time_index(times, meow_ix, meow_size, day_delta=365):
    """Find index in times at least one year from time at meow_ix.
    Args:
        times: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        meow_ix: index into times, used to get day number for comparing
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
    if not enough_time(times, meow_ix, day_delta=365):
        log.debug("insufficient time ({0} days) after times[{1}]:{2}".format(day_delta,meow_ix,times[meow_ix]))
        return None

    end_ix = end_index(meow_ix, meow_size)

    # This seems pretty naive, if you can think of something more
    # performant and elegant, have at it!
    while end_ix < len(times):
        if (times[end_ix]-times[meow_ix]) >= day_delta:
            break
        else:
            end_ix += 1

    log.debug("sufficient time from times[{0}..{1}] (day #{2} to #{3})".format(meow_ix, end_ix, times[meow_ix], times[end_ix]))

    return end_ix


def enough_samples(times, meow_ix, meow_size):
    """Change detection requires a minimum number of samples (as specified
    by meow size).

    This function improves readability of logic that performs this check.

    Args:
        times: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        meow_ix: start index of time/observation window, used with meow_size
            determine if sufficient observations exist.
        meow_size: offset of last time from meow_ix

    Returns:
        bool: True if times contains enough samples after meow_ix,
        False otherwise.
    """
    return (meow_ix+meow_size) <= len(times)


def enough_time(times, meow_ix, day_delta=365):
    """Change detection requires a minimum amount of time (as specified by
    day_delta).

    This function, like `enough_samples` improves readability of logic
    that performs this check.

    Args:
        times: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        meow_ix: start index of time/observation window, used to
            get a value from time for comparison.
        day_delta: minimum difference between time at meow_ix and most
            recent observation.

    Returns:
        list: RMSE for each model.
    """
    return (times[-1]-times[meow_ix]) >= day_delta


def initialize(times, observations, fitter_fn,  model_matrix, tmask_matrix,
               meow_ix, meow_size, adjusted_rmse, day_delta=365):
    """Determine the window indices, models, and errors for observations.

    Args:
        times: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        model_matrix: TODO
        tmask_matrix: TODO
        meow_ix: start index of time/observation window
        meow_size: offset from meow_ix, determines initial window size
        day_delta: minimum difference between time at meow_ix and most
            recent observation

    Returns:
        tuple: start, end, models, errors
    """

    # Guard...
    if not enough_samples(times, meow_ix, meow_size):
        log.debug("failed, insufficient clear observations")
        return meow_ix, None, None, None

    if not enough_time(times, meow_ix, day_delta):
        log.debug("failed, insufficient time range")
        return meow_ix, None, None, None

    while (meow_ix+meow_size) <= len(times):
        log.debug("initialize from {0}..{1}".format(meow_ix, meow_ix+meow_size))

        # Finding a sufficient window of time needs must run
        # each iteration because the starting point (meow_ix)
        # will increment if the model isn't stable, incrementing
        # the window of in lock-step does not guarantee a 1-year+
        # time-range.
        end_ix = find_time_index(times, meow_ix, meow_size, day_delta)
        if end_ix is None:
            break

        # Count outliers in the window, if there are too many outliers then
        # try again.
        times_, observations_ = tmask.tmask(times[meow_ix:end_ix+1],
                                            observations[:,meow_ix:end_ix+1],
                                            tmask_matrix[meow_ix:end_ix+1,:],
                                            adjusted_rmse)

        if (len(times_) < meow_size) or ((times_[-1] - times_[0]) < day_delta):
            log.debug("continue, not enough observations ({0}) after tmask".format(len(times_)))
            meow_ix += 1
            continue

        # Each spectra, although analyzed independently, all share
        # a common time-frame. Consequently, it doesn't make sense
        # to analyze one spectrum in it's entirety.
        period = times[meow_ix:end_ix+1]
        matrix = model_matrix[meow_ix:end_ix+1]
        spectra = observations[:, meow_ix:end_ix+1]
        models = [fitter_fn(period, spectrum) for spectrum in spectra]
        log.debug("update change models")

        # TODO (jmorton): The error of a model is calculated during
        # initialization, but isn't subsequently updated. Determine
        # if this is correct.
        errors_ = rmse(models, matrix, spectra)

        # If a model is not stable, then it is possible that a disturbance
        # exists somewhere in the observation window. The window shifts
        # forward in time, and begins initialization again.
        if not stable(errors_):
            log.debug("unstable model, shift start time and retry")
            meow_ix += 1
            continue
        else:
            log.debug("stable model, done.")
            break

    log.debug("complete, meow_ix: {0}, end_ix: {1}".format(meow_ix, end_ix))
    return meow_ix, end_ix, models, errors_


def extend(times, observations, coefficients,
           meow_ix, end_ix, peek_size, fitter_fn, models):
    """Increase observation window until change is detected.

    Args:
        times: list of ordinal day numbers relative to some epoch,
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

    log.debug("change detection started {0}..{1}".format(meow_ix, end_ix))

    if end_ix is None:
        log.debug("failed, end_ix is None... initialize must have failed")
        return end_ix, models, None

    if (end_ix+peek_size) > len(times):
        log.debug("failed, end_index+peek_size {0}+{1} exceed available data ({2})".format(end_ix, peek_size, len(times)))
        return end_ix, models, None

    while (end_ix+peek_size) <= len(times):
        log.debug("detecting change in times[{0}..{1}]".format(end_ix, end_ix+peek_size))
        peek_ix = end_ix + peek_size

        # TODO (jmorton): Should this be prior and peeked period and spectra
        #      or should this be only the peeked period and spectra?
        time_slice = times[meow_ix:peek_ix]
        coefficient_slice = coefficients[meow_ix:peek_ix]
        spectra_slice = observations[:, meow_ix:peek_ix]

        magnitudes_ = magnitudes(models, coefficient_slice, spectra_slice)
        if accurate(magnitudes_):
            log.debug("no change detected {0}..{1}+{2}".format(meow_ix, end_ix, peek_size))
            models = [fitter_fn(time_slice, spectrum) for spectrum in spectra_slice]
            log.debug("change model updated")
            end_ix += 1
        else:
            log.debug(" change detected, break {0}..{1}+{2}".format(meow_ix, end_ix, peek_size))
            break

    log.debug("change detection finished {0}..{1}".format(meow_ix, end_ix))
    return end_ix, models, magnitudes_


def detect(times, observations, fitter_fn,
           meow_size=16, peek_size=3):
    """Runs the core change detection algorithm.

    The algorithm assumes all pre-processing has been performed on
    observations.

    Args:
        times: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: values for one or more spectra corresponding
            to each time.
        fitter_fn: a function used to fit observation values and
            acquisition dates for each spectra.
        meow_size: minimum expected observation window needed to
            produce a fit.
        peek_size: number of observations to consider when detecting
            a change.

    Returns:
        list: Change models for each observation of each spectra.
    """

    log.debug("build change model – time: {0}, obs: {1}, {2}, meow_size: {3}, peek_size: {4}".format(
              times.shape, observations.shape, fitter_fn, meow_size, peek_size))

    # Accumulator for models. This is a list of lists; each top-level list
    # corresponds to a particular spectra.
    results = ()

    # The starting point for initialization. Used to as reference point for
    # taking a range of times and spectral values.
    meow_ix = 0

    # calculate the adjusted RMSE
    adjusted_rmse = np.median(np.absolute(observations), 1) * app.T_CONST

    # pre-calculate coefficient matrix for all time values; this calculation
    # needs to be performed only once, but the lasso and tmask matrices are
    # different.
    model_matrix = lasso.coefficient_matrix(times)
    tmask_matrix = tmask.robust_fit_coefficient_matrix(times)

    # Only build models as long as sufficient data exists. The observation
    # window starts at meow_ix and is fixed until the change model no longer
    # fits new observations, i.e. a change is detected. The meow_ix updated
    # at the end of each iteration using an end index, so it is possible
    # it will become None.
    while (meow_ix is not None) and (meow_ix+meow_size) <= len(times):

        # Step 1: Initialize -- find an initial stable time-frame.
        log.debug("initialize change model")
        meow_ix, end_ix, models, errors_ = initialize(times, observations,
                                                      fitter_fn, model_matrix, tmask_matrix,
                                                      meow_ix, meow_size,
                                                      adjusted_rmse)

        # Step 2: Extension -- expand time-frame until a change is detected.
        log.debug("extend change model")
        end_ix, models, magnitudes_ = extend(times, observations, model_matrix,
                                             meow_ix, end_ix, peek_size,
                                             fitter_fn, models)

        # After initialization and extension, the change models for each
        # spectra are complete for a period of time. If meow_ix and end_ix
        # are not present, then not enough observations exist for a useful
        # model to be produced, so nothing is appened to results.
        log.debug("accumulate results: {} so far".format(len(results)))
        if (meow_ix is not None) and (end_ix is not None):
            result = (times[meow_ix], times[end_ix],
                      models, errors_, magnitudes_)
            results += (result,)

        # Step 4: Iterate. The meow_ix is moved to the end of the current
        # timeframe and a new model is generated. It is possible for end_ix
        # to be None, in which case iteration stops.
        meow_ix = end_ix

    log.debug("change detection complete")
    return results
