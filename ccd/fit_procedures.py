import numpy as np

from ccd import tmask
from ccd.models import lasso
from ccd.app import logging, config


log = logging.getLogger(__name__)


def permanent_snow_procedure():
    pass


def fmask_fail_procedure():
    pass


def standard_fit_procedure(times, observations, fitter_fn,
                           meow_size=config.MEOW_SIZE, peek_size=config.PEEK_SIZE):
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

    log.debug("build change model – time: {0}, obs: {1}, {2}, \
                   meow_size: {3}, peek_size: {4}".format(
        times.shape, observations.shape,
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
    model_matrix = lasso.coefficient_matrix(times)
    tmask_matrix = tmask.robust_fit_coefficient_matrix(times)

    # Only build models as long as sufficient data exists. The observation
    # window starts at meow_ix and is fixed until the change model no longer
    # fits new observations, i.e. a change is detected. The meow_ix updated
    # at the end of each iteration using an end index, so it is possible
    # it will become None.
    while (meow_ix is not None) and (meow_ix + meow_size) <= len(times):

        # Step 1: Initialize -- find an initial stable time-frame.
        log.debug("initialize change model")
        meow_ix, end_ix, models, errors_ = initialize(times, observations,
                                                      fitter_fn, model_matrix,
                                                      tmask_matrix,
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
        if (meow_ix is not None) and (end_ix is not None):
            result = (times[meow_ix], times[end_ix],
                      models, errors_, magnitudes_)
            results += (result,)

        log.debug("accumulate results, {} so far".format(len(results)))
        # Step 4: Iterate. The meow_ix is moved to the end of the current
        # timeframe and a new model is generated. It is possible for end_ix
        # to be None, in which case iteration stops.
        meow_ix = end_ix

    log.debug("change detection complete")
    return results
