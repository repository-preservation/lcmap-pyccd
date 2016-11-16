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

from ccd import tmask, qa
from ccd.models import lasso
from ccd.app import logging, config
from ccd.change import initialize, extend


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


def permanent_snow_procedure():
    pass


def fmask_fail_procedure():
    pass


def standard_fit_procedure(dates, observations, fitter_fn,
                           meow_size=config.MEOW_SIZE, peek_size=config.PEEK_SIZE):
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

    # Accumulator for models. This is a list of lists; each top-level list
    # corresponds to a particular spectra.
    results = ()

    # The starting point for initialization. Used to as reference point for
    # taking a range of times and spectral values.
    meow_ix = 0

    # calculate a modified first-order variogram/madogram
    adjusted_rmse = np.median(np.abs(np.diff(observations, n=1, axis=1)), axis=1)
    # ...or is this correct?
    # adjusted_rmse = np.median(np.absolute(observations), 1) * app.T_CONST

    # pre-calculate coefficient matrix for all time values; this calculation
    # needs to be performed only once, but the lasso and tmask matrices are
    # different.
    model_matrix = lasso.coefficient_matrix(dates)
    tmask_matrix = tmask.tmask_coefficient_matrix(dates)

    # Only build models as long as sufficient data exists. The observation
    # window starts at meow_ix and is fixed until the change model no longer
    # fits new observations, i.e. a change is detected. The meow_ix updated
    # at the end of each iteration using an end index, so it is possible
    # it will become None.
    while (meow_ix is not None) and (meow_ix + meow_size) <= len(dates):

        # Step 1: Initialize -- find an initial stable time-frame.
        log.debug("initialize change model")
        meow_ix, end_ix, models, errors_ = initialize(dates, observations,
                                                      fitter_fn, model_matrix,
                                                      tmask_matrix,
                                                      meow_ix, meow_size,
                                                      adjusted_rmse)

        # Step 2: Extension -- expand time-frame until a change is detected.
        log.debug("extend change model")
        end_ix, models, magnitudes_ = extend(dates, observations, model_matrix,
                                             meow_ix, end_ix, peek_size,
                                             fitter_fn, models)

        # After initialization and extension, the change models for each
        # spectra are complete for a period of time. If meow_ix and end_ix
        # are not present, then not enough observations exist for a useful
        # model to be produced, so nothing is appened to results.
        if (meow_ix is not None) and (end_ix is not None):
            result = (dates[meow_ix], dates[end_ix],
                      models, errors_, magnitudes_)
            results += (result,)

        log.debug("accumulate results, {} so far".format(len(results)))
        # Step 4: Iterate. The meow_ix is moved to the end of the current
        # timeframe and a new model is generated. It is possible for end_ix
        # to be None, in which case iteration stops.
        meow_ix = end_ix

    log.debug("change detection complete")
    return results
