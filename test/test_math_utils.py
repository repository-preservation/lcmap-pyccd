from ccd.math_utils import *


def test_euclidean_norm():
    arr = np.arange(5)
    ans = 30.0 ** .5

    assert ans == euclidean_norm(arr)


def test_sum_of_squares():
    arr = np.arange(5)
    ans = 30

    assert ans == sum_of_squares(arr)

    arr2d = np.arange(10).reshape(2, -1)
    ans2d = [30, 255]

    assert np.array_equal(ans2d, sum_of_squares(arr2d, 1))


def test_calc_rmse():
    actual = np.arange(5)
    pred = np.arange(5) + 1

    ans_resids = np.ones_like(actual) * -1
    ans_rmse = 1.0

    rmse, resids = calc_rmse(actual, pred)

    assert rmse == ans_rmse
    assert np.array_equal(ans_resids, resids)


def test_calc_median():
    arr = np.arange(5)
    ans = 2

    assert ans == calc_median(arr)


def test_calc_residuals():
    actual = np.arange(5)
    pred = np.arange(5) + 1

    ans_resids = np.ones_like(actual) * -1

    assert np.array_equal(ans_resids, calc_residuals(actual, pred))


def test_kelvin_to_celsius():
    pass
