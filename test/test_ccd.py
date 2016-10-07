import numpy as np

from shared import acquisition_delta
from shared import sinusoid
from shared import sample_data

from ccd.models import lasso
import ccd.change as change


def test_not_enough_observations():
    times, observations = sample_data('R15/2000-01-01/P16D')
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 0


def test_one_year_minimum():
    times, observations = sample_data('R16/2000-01-01/P1W')
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    time_delta = times[-1]-times[1]
    assert time_delta < 365
    assert len(models) == 0


def test_enough_observations():
    times, observations = sample_data('R16/2000-01-01/P1M')
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    time_delta = times[-1]-times[0]
    assert time_delta > 365
    assert len(models) == 1


def test_change_windows(n=50, meow_size=16, peek_size=3):
    times, observations = sample_data('R{0}/2000-01-01/P16D'.format(n))
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn,
                           meow_size=meow_size, peek_size=peek_size)
    time_delta = times[-1]-times[0]
    assert time_delta > 365
    assert len(models) == 1


def testing_an_unstable_initial_period():
    times, observations = sample_data('R50/2000-01-01/P16D')
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 1


def test_two_changes_during_time():
    times, observations = sample_data('R100/2000-01-01/P1M')
    two_periods = np.hstack((sinusoid(times[0:50])+10, sinusoid(times[50:100])+50))
    observations[0] = two_periods
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 2


def test_three_changes_during_time():
    times, observations = sample_data('R150/2000-01-01/P32D')
    three_periods = np.hstack((sinusoid(times[0:50]) + 10,
                               sinusoid(times[50:100]) + 50,
                               sinusoid(times[100:150]) + 10))
    observations[0] = three_periods
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 3
