# import numpy as np
#
# from shared import sinusoid
# from shared import sample_line, sample_sinusoid
#
# from ccd.models import lasso
# import ccd.procedures as change


# def test_not_enough_observations():
#     times, observations = sample_line('R15/2000-01-01/P16D')
#     fitter_fn = lasso.fitted_model
#     quality = np.zeros(shape=times.shape)
#
#     models, mask = change.standard_procedure(times, observations, fitter_fn, quality)
#
#     assert len(models) == 1
#
#
# def test_one_year_minimum():
#     times, observations = sample_line('R16/2000-01-01/P1W')
#     fitter_fn = lasso.fitted_model
#     quality = np.zeros(shape=times.shape)
#
#     models, mask = change.standard_procedure(times, observations, fitter_fn, quality)
#     time_delta = times[-1]-times[1]
#
#     assert time_delta < 365
#     assert len(models) == 1
#
#
# def test_enough_observations():
#     times, observations = sample_line('R18/2000-01-01/P1M')
#     fitter_fn = lasso.fitted_model
#     quality = np.zeros(shape=times.shape)
#
#     models, mask = change.standard_procedure(times, observations, fitter_fn, quality)
#     time_delta = times[-1]-times[0]
#
#     assert time_delta > 365
#     assert len(models) == 1
#
#
# def test_change_windows(n=50, meow_size=16, peek_size=3):
#     times, observations = sample_line('R{0}/2000-01-01/P16D'.format(n))
#     fitter_fn = lasso.fitted_model
#     quality = np.zeros(shape=(observations.shape[1],))
#     models = change.standard_procedure(times, observations, fitter_fn, quality,
#                                            meow_size=meow_size, peek_size=peek_size)
#     time_delta = times[-1]-times[0]
#     assert time_delta > 365
#     assert len(models) == 1, models
#
#
# def testing_an_unstable_initial_period():
#     times, observations = sample_line('R50/2000-01-01/P16D')
#     fitter_fn = lasso.fitted_model
#     quality = np.zeros(shape=(observations.shape[1],))
#     models = change.standard_procedure(times, observations, fitter_fn, quality)
#     assert len(models) == 1
#
#
# def test_two_changes_during_time():
#     times, observations = sample_line('R100/2000-01-01/P16D')
#     observations[0, 50:100] += 500
#     fitter_fn = lasso.fitted_model
#     quality = np.zeros(shape=(observations.shape[1],))
#     models = change.standard_procedure(times, observations, fitter_fn, quality)
#     length = len(models)
#     # TODO (jmorton) Figure out how to generate sample data
#     #      that doesn't confuse change detection.
#     # assert length == 2, length
#
#
# def test_three_changes_during_time():
#     times, observations = sample_line('R150/2000-01-01/P32D')
#     quality = np.zeros(shape=(observations.shape[1],))
#     three_periods = np.hstack((sinusoid(times[0:50]) + 10,
#                                sinusoid(times[50:100]) + 500,
#                                sinusoid(times[100:150]) + 10))
#     observations[0] = three_periods
#     fitter_fn = lasso.fitted_model
#     models = change.standard_procedure(times, observations, fitter_fn, quality)
#     # TODO (jmorton) Figure out how to generate sample data
#     #      that doesn't confuse change detection.
#     # assert len(models) == 3, len(models)
