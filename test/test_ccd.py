import numpy as np

from shared import acquisition_delta
from shared import sinusoid

from ccd.models import lasso
import ccd.change as change


def test_not_enough_observations():
    times = acquisition_delta('R15/2000-01-01/P16D')
    reds = sinusoid(times)
    observations = np.array([reds])
    fitter_fn = lasso.fitted_model
    # pytest.set_trace()
    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 0


def test_one_year_minimum():
    times = acquisition_delta('R16/2000-01-01/P1W')
    reds = sinusoid(times)
    observations = np.array([reds])
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    time_delta = times[-1]-times[1]
    assert time_delta < 365
    assert len(models) == 0


def test_enough_observations():
    times = acquisition_delta('R16/2000-01-01/P1M')
    reds = sinusoid(times)
    blues = sinusoid(times)
    greens = sinusoid(times)
    observations = np.array([reds, blues, greens])
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn)
    time_delta = times[-1]-times[1]
    assert time_delta > 365
    assert len(models) == 1


def test_change_windows(n=50, meow_size=16, peek_size=3):
    times = acquisition_delta('R{0}/2000-01-01/P16D'.format(n))
    reds = sinusoid(times)
    greens = sinusoid(times)
    blues = sinusoid(times)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model
    models = change.detect(times, observations, fitter_fn,
                           meow_size=meow_size, peek_size=peek_size)
    # If we only accumulate stable models...
    expected = 1
    # If we accumulate all steps...
    # expected = n - meow_size - peek_size + 2
    time_delta = times[-1]-times[1]
    assert time_delta > 365
    assert len(models) == expected


def testing_an_unstable_initial_period():
    times = acquisition_delta('R50/2000-01-01/P16D')
    reds = np.hstack((sinusoid(times[0:10])+10, sinusoid(times[10:50])+50))
    blues = np.hstack((sinusoid(times[0:10])+10, sinusoid(times[10:50])+50))
    greens = np.hstack((sinusoid(times[0:10])+10, sinusoid(times[10:50])+50))
    observations = np.array([reds, blues, greens])
    fitter_fn = lasso.fitted_model

    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 1


def test_two_changes_during_time():
    times = acquisition_delta('R100/2000-01-01/P1M')
    reds = np.hstack((sinusoid(times[0:50])+10, sinusoid(times[50:100])+50))
    greens = sinusoid(times)
    blues = sinusoid(times)
    observations = np.array([reds, greens, blues])
    fitter_fn = lasso.fitted_model

    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 2


def test_three_changes_during_time():
    times = acquisition_delta('R150/2000-01-01/P32D')
    reds = np.hstack((sinusoid(times[0:50]) + 10,
                      sinusoid(times[50:100]) + 50,
                      sinusoid(times[100:150]) + 10))
    observations = np.array([reds])
    fitter_fn = lasso.fitted_model

    models = change.detect(times, observations, fitter_fn)
    assert len(models) == 3


def test_T500_S10():
    times = acquisition_delta('R500/1990-01-01/P8D')
    reds = np.hstack((sinusoid(times[0:50]) + 10,
                      sinusoid(times[50:100]) + 50,
                      sinusoid(times[100:150]) + 10,
                      sinusoid(times[150:200]) + 50,
                      sinusoid(times[200:250]) + 10,
                      sinusoid(times[200:250]) + 50,
                      sinusoid(times[250:300]) + 10,
                      sinusoid(times[300:350]) + 50,
                      sinusoid(times[350:400]) + 10,
                      sinusoid(times[400:450]) + 50,
                      sinusoid(times[450:500]) + 10))
    observations = np.array([reds])
    fitter_fn = lasso.fitted_model

    models = change.detect(times, observations, fitter_fn)
    assert len(models) > 0


def test_T500_S1():
    times = acquisition_delta('R500/1990-01-01/P8D')
    reds = np.hstack((sinusoid(times[0:500])+10))
    observations = np.array([reds])
    fitter_fn = lasso.fitted_model

    models = change.detect(times, observations, fitter_fn)
    assert len(models) > 0
