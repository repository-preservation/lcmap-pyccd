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
from ccd.change import initialize, extend, lookback, change_magnitudes, update_processing_mask
from ccd.models import lasso, tmask
from ccd.math_utils import kelvin_to_celsius


log = logging.getLogger(__name__)


def determine_fit_procedure(quality):
    """Determine which curve fitting function to use

    This is based on information from the QA band

    Args:
        quality: QA information for each observation

    Returns:
        method: the corresponding method that will be use to generate the curves
    """
    if not qa.enough_clear(quality):
        if qa.enough_snow(quality):
            return permanent_snow_procedure
        else:
            return fmask_fail_procedure
    else:
        return standard_fit_procedure


def permanent_snow_procedure(dates, observations, fitter_fn, quality):
    pass


def fmask_fail_procedure(dates, observations, fitter_fn, quality):
    pass


def standard_fit_procedure(dates, observations, fitter_fn, quality,
                           meow_size=defaults.MEOW_SIZE, peek_size=defaults.PEEK_SIZE,
                           thermal_idx=defaults.THERMAL_IDX):
    """Runs the core change detection algorithm.

        The algorithm assumes all pre-processing has been performed on
        observations.

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
        """

    log.debug("build change model – time: {0}, obs: {1}, {2}, \
                   meow_size: {3}, peek_size: {4}".format(
        dates.shape, observations.shape,
        fitter_fn, meow_size, peek_size))

    # First we need to filter the observations based on the spectra values
    # and qa information and convert kelvin to celsius
    # We then persist the processing mask through subsequent operations as
    # additional data points get identified to be excluded from processing
    observations[thermal_idx] = kelvin_to_celsius(observations[thermal_idx])
    processing_mask = qa.standard_filter(observations, quality)

    # All we care about now is stuff that passed the filtering and we
    # need to convert the thermal values
    # dates = dates[filter_idxs]
    # observations = observations[:, filter_idxs]
    # quality = quality[filter_idxs]


    # Accumulator for models. This is a list of lists; each top-level list
    # corresponds to a particular spectra.
    results = tuple()

    # Initialize the window which is used for building the models
    # this can actually be different than the start and ending indices
    # that are used for the time-span that the model covers
    # thus we need to initialize a starting index value as well
    model_window = slice(0, meow_size)
    start_ix = 0

    # calculate a modified first-order variogram/madogram
    # adjusted_rmse = np.median(np.abs(np.diff(observations)), axis=1)

    # pre-calculate coefficient matrix for all time values; this calculation
    # needs to be performed only once, we can not do this for the
    # lasso regression coefficients as the number of coefficients changes
    # based on the number of observations that are fed into it increases.
    # model_matrix = lasso.coefficient_matrix(dates)
    tmask_matrix = tmask.tmask_coefficient_matrix(dates)

    # Only build models as long as sufficient data exists. The observation
    # window starts at meow_ix and is fixed until the change model no longer
    # fits new observations, i.e. a change is detected. The meow_ix updated
    # at the end of each iteration using an end index, so it is possible
    # it will become None.
    while model_window.stop <= dates.shape[0] - meow_size:

        # Step 1: Initialize -- find an initial stable time-frame.
        log.debug("initialize change model")
        model_window, models = initialize(dates, observations, fitter_fn,
                                          tmask_matrix, model_window,
                                          meow_size, processing_mask)

        if model_window.start > start_ix:
            # TODO look at past the difference in indicies to see if they
            # fall into the initialized model
            model_window, outliers = lookback(dates, observations,
                                              model_window, peek_size, models,
                                              start_ix, processing_mask)
            processing_mask = update_processing_mask(processing_mask, outliers)

        # If we are at the beginning of the time series and if initialize
        # has moved forward the start of the first curve by more than the
        # peek size, then we should fit a general curve to those first
        # spectral values
        if not results and model_window.start - peek_size > start_ix:
            # TODO make uniform method for fitting models and returning the
            # appropriate information
            # Maybe define a namedtuple for model storage
            models_tmp = [fitter_fn(dates[processing_mask][start_ix:model_window.start],
                                    spectrum)
                          for spectrum
                          in observations[:, processing_mask][:, start_ix:model_window.start]]

            mag_tmp = change_magnitudes(dates[processing_mask][start_ix:model_window.start],
                                        observations[processing_mask][start_ix:model_window.start],
                                        models_tmp)

            results += (dates[model_window.start], dates[model_window.stop],
                        models_tmp, mag_tmp)

        # Step 2: Extension -- expand time-frame until a change is detected.
        # initialized models from Step 1 and the lookback
        # cannot be passed along due to how
        # Tmask can throw out some values used in that model, but are
        # subsequently used in follow on methods
        log.debug("extend change model")
        model_window, models, magnitudes_, outliers = extend(dates, observations,
                                                             model_window, peek_size,
                                                             fitter_fn, processing_mask)
        processing_mask = update_processing_mask(processing_mask, outliers)

        # After initialization and extension, the change models for each
        # spectra are complete for a period of time. If meow_ix and end_ix
        # are not present, then not enough observations exist for a useful
        # model to be produced, so nothing is appened to results.
        if (model_window.start is not None) and (model_window.stop is not None):
            result = (dates[processing_mask][model_window.start],
                      dates[processing_mask][model_window.stop],
                      models, magnitudes_)
            results += (result,)

        log.debug("accumulate results, {} so far".format(len(results)))
        # Step 4: Iterate. The meow_ix is moved to the end of the current
        # timeframe and a new model is generated. It is possible for end_ix
        # to be None, in which case iteration stops.
        start_ix = model_window.stop
        model_window = slice(model_window.stop, model_window.stop + meow_size)

    log.debug("change detection complete")
    return results
