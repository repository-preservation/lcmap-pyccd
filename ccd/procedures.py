"""Functions for providing the over-arching methodology. Tying together the
individual components that make-up the change detection process.

The results of this process is a list-of-lists of change models that correspond
to observation spectra. A processing mask is also returned, outlining which
observations were utilized and which were not.

Pre-processing routines are essential to, but distinct from, the core change
detection algorithm. See the `ccd.qa` for more details related to this
step.

For more information please refer to the `pyccd Algorithm Description Document`.

.. _Algorithm Description Document:
   https://drive.google.com/drive/folders/0BzELHvbrg1pDREJlTF8xOHBZbEU
"""

import numpy as np

from ccd import qa
from ccd.app import logging, defaults
from ccd.change import initialize, lookforward, lookback, catch
from ccd.models import results_to_changemodel
from ccd.math_utils import kelvin_to_celsius, calculate_variogram


log = logging.getLogger(__name__)


def fit_procedure(quality):
    """Determine which curve fitting method to use

    This is based on information from the QA band

    Args:
        quality: QA information for each observation

    Returns:
        method: the corresponding method that will be use to generate the curves
    """
    if not qa.enough_clear(quality):
        if qa.enough_snow(quality):
            func = permanent_snow_procedure
        else:
            func = insufficient_clear_procedure
    else:
        func = standard_procedure

    log.debug('Procedure selected: %s',
              func.__name__)

    return func


def permanent_snow_procedure(dates, observations, fitter_fn, quality,
                             meow_size=defaults.MEOW_SIZE):
    """
    Snow procedure for when there is a significant amount snow represented
    in the quality information

    This method essentially fits a 4 coefficient model across all the
    observations

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: values for one or more spectra corresponding
            to each time.
        fitter_fn: a function used to fit observation values and
            acquisition dates for each spectra.
        quality: QA information for each observation
        meow_size: minimum expected observation window needed to
            produce a fit.

    Returns:
        list: Change models for each observation of each spectra.
        1-d ndarray: processing mask indicating which values were used
            for model fitting
    """
    processing_mask = qa.snow_procedure_filter(observations, quality)

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    if np.sum(processing_mask) < meow_size:
        return [], processing_mask

    models = [fitter_fn(period, spectrum, 4)
              for spectrum in spectral_obs]

    magnitudes = np.zeros(shape=(observations.shape[0],))

    # White space is cheap, so let's use it
    result = results_to_changemodel(fitted_models=models,
                                    start_day=dates[0],
                                    end_day=dates[-1],
                                    break_day=0,
                                    magnitudes=magnitudes,
                                    observation_count=np.sum(processing_mask),
                                    change_probability=0,
                                    num_coefficients=4)

    return (result,), processing_mask


def insufficient_clear_procedure(dates, observations, fitter_fn, quality,
                                 meow_size=defaults.MEOW_SIZE):
    """
    insufficient clear procedure for when there is an insufficient quality
    observations

    This method essentially fits a 4 coefficient model across all the
    observations

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: values for one or more spectra corresponding
            to each time.
        fitter_fn: a function used to fit observation values and
            acquisition dates for each spectra.
        quality: QA information for each observation
        meow_size: minimum expected observation window needed to
            produce a fit.

    Returns:
        list: Change models for each observation of each spectra.
        1-d ndarray: processing mask indicating which values were used
            for model fitting
        """
    processing_mask = qa.insufficient_clear_filter(observations, quality)

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    if np.sum(processing_mask) < meow_size:
        return [], processing_mask

    models = [fitter_fn(period, spectrum, 4)
              for spectrum in spectral_obs]

    magnitudes = np.zeros(shape=(observations.shape[0],))

    result = results_to_changemodel(fitted_models=models,
                                    start_day=dates[0],
                                    end_day=dates[-1],
                                    break_day=0,
                                    magnitudes=magnitudes,
                                    observation_count=np.sum(processing_mask),
                                    change_probability=0,
                                    num_coefficients=4)

    return (result,), processing_mask


def standard_procedure(dates, observations, fitter_fn, quality,
                       meow_size=defaults.MEOW_SIZE,
                       peek_size=defaults.PEEK_SIZE,
                       thermal_idx=defaults.THERMAL_IDX):
    """
    Runs the core change detection algorithm.

    Step 1: initialize -- Find an initial stable time-frame to build from.

    Step 2: lookback -- The initlize step may have iterated the start of the
    model past the previous break point. If so then we need too look back at
    previous values to see if they can be included within the new
    initialized model.

    Step 3: catch -- Fit a general model to values that may have been skipped
    over by the previous steps.

    Step 4: lookforward -- Expand the time-frame until a change is detected.

    Step 5: Iterate.

    Step 6: catch -- End of time series considerations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: 2-d array of observed spectral values corresponding
            to each time.
        fitter_fn: a function used to fit observation values and
            acquisition dates for each spectra.
        quality: QA information for each observation
        meow_size: minimum expected observation window needed to
            produce a fit.
        peek_size: number of observations to consider when detecting
            a change.
        thermal_idx: index location of the thermal band in the observation
            2-d array

    Returns:
        list: Change models for each observation of each spectra.
        1-d ndarray: processing mask indicating which values were used
            for model fitting
    """

    log.debug('Build change models - dates: %s, obs: %s, '
              'meow_size: %s, peek_size: %s',
              dates.shape[0], observations.shape, meow_size, peek_size)

    # First we need to filter the observations based on the spectra values
    # and qa information and convert kelvin to celsius.
    # We then persist the processing mask through subsequent operations as
    # additional data points get identified to be excluded from processing.
    observations[thermal_idx] = kelvin_to_celsius(observations[thermal_idx])

    # There's two ways to handle the boolean mask with the windows in
    # subsequent processing:
    # 1. Apply the mask and adjust the window values to compensate for the
    # values that are taken out.
    # 2. Apply the window to the data and use the mask only for that window
    # Option 2 allows window values to be applied directly to the input data.
    # But now you must compensate when you want to have certain sized windows
    # which brings other complications in the iterative steps.

    # The masked module from numpy does not seem to really add anything of
    # benefit to what we need to do, plus scikit may still be incompatible
    # with them.
    processing_mask = qa.standard_procedure_filter(observations, quality)

    obs_count = np.sum(processing_mask)

    log.debug('Processing mask initial count: %s', obs_count)

    # Accumulator for models. This is a list of ChangeModel named tuples
    results = []

    if obs_count <= meow_size:
        return results, processing_mask

    # Initialize the window which is used for building the models
    model_window = slice(0, meow_size)
    previous_end = 0

    # Calculate the variogram/madogram that will be used in subsequent
    # processing steps. See algorithm documentation for further information.
    variogram = calculate_variogram(observations[:, processing_mask])
    log.debug('Variogram values: %s', variogram)

    # Only build models as long as sufficient data exists.
    while model_window.stop <= dates[processing_mask].shape[0] - meow_size:
        # Step 1: Initialize
        log.debug('Initialize for change model #: %s', len(results) + 1)

        model_window, init_models, processing_mask = initialize(dates,
                                                                observations,
                                                                fitter_fn,
                                                                model_window,
                                                                meow_size,
                                                                peek_size,
                                                                processing_mask,
                                                                variogram)

        # Catch for failure
        if init_models is None:
            log.debug('Model initialization failed')
            break

        # Step 2: Lookback
        if model_window.start > previous_end:
            model_window, processing_mask = lookback(dates,
                                                     observations,
                                                     model_window,
                                                     peek_size,
                                                     init_models,
                                                     previous_end,
                                                     processing_mask,
                                                     variogram)

        # Step 3: catch
        # If we have moved > peek_size from the previous break point
        # then we fit a generalized model to those points.
        if model_window.start - previous_end > peek_size:
                results.append(catch(dates,
                                     observations,
                                     fitter_fn,
                                     processing_mask,
                                     slice(previous_end, model_window.start)))

        # Step 4: lookforward
        log.debug('Extend change model')
        result, processing_mask, model_window = lookforward(dates,
                                                            observations,
                                                            model_window,
                                                            peek_size,
                                                            fitter_fn,
                                                            processing_mask,
                                                            variogram)
        results.append(result)

        log.debug('Accumulate results, {} so far'.format(len(results)))

        # Step 5: Iterate
        previous_end = model_window.stop
        model_window = slice(model_window.stop, model_window.stop + meow_size)

    # Step 6: Catch
    # We can use previous start here as that value should be equal to
    # model_window.stop due to the constraints on the the previous while
    # loop.
    if previous_end + peek_size < dates[processing_mask].shape[0]:
        model_window = slice(previous_end, dates[processing_mask].shape[0])
        results.append(catch(dates, observations, fitter_fn,
                             processing_mask, model_window))

    log.debug("change detection complete")

    return results, processing_mask
