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

from ccd import qa
from ccd.app import logging, defaults
from ccd.change import initialize, lookforward, lookback, change_magnitude, update_processing_mask, catch
from ccd.models import lasso, tmask, SpectralModel, ChangeModel, results_to_changemodel
from ccd.math_utils import kelvin_to_celsius, calculate_variogram


log = logging.getLogger(__name__)


class ProcedureException(Exception):
    pass


def fit_procedure(quality):
    """Determine which curve fitting function to use

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
                             meow_size=defaults.MEOW_SIZE,
                             peek_size=defaults.PEEK_SIZE,
                             thermal_idx=defaults.THERMAL_IDX):
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
        meow_size: minimum expected observation window needed to
            produce a fit.
        peek_size: number of observations to consider when detecting
            a change.

    Returns:

    """
    processing_mask = qa.snow_procedure_filter(observations, quality)

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    if np.sum(processing_mask) < meow_size:
        raise ProcedureException('Insufficient snow/water/clear '
                                 'observations for the snow procedure')

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
                                 meow_size=defaults.MEOW_SIZE,
                                 peek_size=defaults.PEEK_SIZE,
                                 thermal_idx=defaults.THERMAL_IDX):
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
        meow_size: minimum expected observation window needed to
            produce a fit.
        peek_size: number of observations to consider when detecting
            a change.

    Returns:

        """
    processing_mask = qa.insufficient_clear_filter(observations, quality)

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    if np.sum(processing_mask) < meow_size:
        raise ProcedureException('Insufficient clear observations for the '
                                 'insufficient_clear_procedure')

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
                       thermal_idx=defaults.THERMAL_IDX,
                       day_delta=defaults.DAY_DELTA):
    """
    Runs the core change detection algorithm.
    Step 1: Initialize -- find an initial stable time-frame.

    Step 2: Lookback -- we need too look back at previous values to see
    if they can be included with the new initialized model

    Step 3: lookforward -- expand time-frame until a change is detected.
    initialized models from Step 1 and the lookback cannot be passed
    along due to how Tmask can throw out some values used in that model,
    but are subsequently used in follow on methods

    Step 4: Iterate. The previous_start is moved to the end of the current
    timeframe and a new model is generated. It is possible for end_ix
    to be None, in which case iteration stops.

    Step 5: Catch. End of time series considerations, also provides for
    building models for short sets of data

    Args:
        dates: list of ordinal day numbers relative to some epoch,
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
        1-d ndarray: processing mask indicating which values were used
            for model fitting
    """

    log.debug('Build change models – dates: %s, obs: %s, '
              'meow_size: %s, peek_size: %s',
              dates.shape[0], observations.shape, meow_size, peek_size)

    # First we need to filter the observations based on the spectra values
    # and qa information and convert kelvin to celsius
    # We then persist the processing mask through subsequent operations as
    # additional data points get identified to be excluded from processing
    observations[thermal_idx] = kelvin_to_celsius(observations[thermal_idx])

    # There's two ways to handle the boolean mask with the windows in
    # subsequent processing:
    # 1. Apply the mask and adjust the window values to compensate for the
    # values that are taken out.
    # 2. Apply the window to the data and use the mask only for that window
    # Option 2 allows window values to be applied directly to the input data.
    # But now you must compensate when you want to have certain sized windows.
    # This would also seem to be important for subsequent analysis
    # utilizing the logs.

    # The masked module from numpy does not seem to really add anything of
    # benefit to what we need to do, plus scikit may still be incompatible
    # with them.
    processing_mask = qa.standard_procedure_filter(observations, quality)

    obs_count = np.sum(processing_mask)

    log.debug('Processing mask initial count: %s', obs_count)

    if obs_count <= peek_size:
        raise ValueError('Insufficient data available after initial masking')

    # Accumulator for models. This is a list of ChangeModel named tuples
    results = []

    # Initialize the window which is used for building the models
    # this can actually be different than the start and ending indices
    # that are used for the time-span that the model covers
    # thus we need to initialize a starting index value as well
    model_window = slice(0, meow_size)
    previous_start = 0

    variogram = calculate_variogram(observations[:, processing_mask])
    log.debug('Variogram values: %s', variogram)

    # Only build models as long as sufficient data exists.
    while model_window.stop <= dates[processing_mask].shape[0] - meow_size:
        # Step 1: Initialize
        log.debug('Initialize for change model #: %s', len(results) + 1)
        # Hopefully, one of these days line lengths for pep8 will change,
        # and I won't have to worry about people saying something about it
        model_window, init_models, processing_mask = initialize(dates,
                                                                observations,
                                                                fitter_fn,
                                                                model_window,
                                                                meow_size,
                                                                peek_size,
                                                                processing_mask,
                                                                variogram)

        if init_models is None:
            log.debug('Model initialization failed')
            break

        # Step 2: Lookback
        if model_window.start > previous_start:
            model_window, processing_mask = lookback(dates,
                                                     observations,
                                                     model_window,
                                                     peek_size,
                                                     init_models,
                                                     previous_start,
                                                     processing_mask,
                                                     variogram)

        # If we are at the beginning of the time series and if initialize
        # has moved forward the start of the first curve by more than the
        # peek size, then we should fit a general curve to those first
        # spectral values.
        if not results and model_window.start - peek_size > 0:
            results.append(catch(dates, observations, fitter_fn,
                                 processing_mask, slice(0, model_window.start)))

        # Step 3: lookforward
        log.debug('Extend change model')
        result, processing_mask = lookforward(dates,
                                              observations,
                                              model_window,
                                              peek_size,
                                              fitter_fn,
                                              processing_mask,
                                              variogram)
        results.append(result)

        log.debug('Accumulate results, {} so far'.format(len(results)))

        # Step 4: Iterate
        previous_start = model_window.stop
        model_window = slice(model_window.stop, model_window.stop + meow_size)

    # Step 5: Catch
    # We can use previous start here as that value should be equal to
    # model_window.stop due to the constraints on the the previous while
    # loop.
    if previous_start + peek_size < dates[processing_mask].shape[0]:
        model_window = slice(previous_start, dates[processing_mask].shape[0])
        results.append(catch(dates, observations, fitter_fn,
                             processing_mask, model_window))

    log.debug("change detection complete")

    return results, processing_mask
