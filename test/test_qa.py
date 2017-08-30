"""
Tests for the basic masking and filtering operations
"""
from ccd.qa import *
from ccd.app import get_default_params


clear = 0
water = 1
fill = 255
snow = 3
clear_thresh = 0.25
snow_thresh = 0.75


default_params = get_default_params()


def test_checkbit():
    packint = 1
    offset = 0

    assert checkbit(packint, offset)

    offset = 1

    assert not checkbit(packint, offset)


def test_qabitval():
    # [fill, clear, water, cloud shadow, snow, cloud,
    # High Cirrus + low cloud conf, high cirrus + medium cloud conf, terrain occlusion]
    packints = [1, 2, 4, 8, 16, 32, 832, 896, 1024]
    ans = [0, 1, 2, 3, 4, 5, 1, 1, 1]

    for i, a in zip(packints, ans):
        assert qabitval(i, default_params) == a


def test_count_clear_or_water():
    arr = np.arange(5)
    ans = 2

    assert ans == count_clear_or_water(arr, clear, water)


def test_count_total():
    arr = np.arange(5)
    arr[-1] = 255
    ans = 4

    assert ans == count_total(arr, fill)


def test_ratio_clear():
    arr = np.arange(5)
    arr[-1] = 255
    ans = 0.5

    assert ans == ratio_clear(arr, clear, water, fill)


def test_ratio_snow():
    arr = np.arange(5)
    ans = 1/3.01

    assert ans == ratio_snow(arr, clear, water, snow)


def test_enough_clear():
    arr = np.arange(5)
    ans = True

    assert ans == enough_clear(arr, clear, water, fill, clear_thresh)

    arr[1:] = snow
    ans = False
    assert ans == enough_clear(arr, clear, water, fill, clear_thresh)


def test_enough_snow():
    arr = np.arange(5)
    ans = False

    assert ans == enough_snow(arr, clear, water, snow, snow_thresh)

    arr[1:] = snow
    ans = True
    assert ans == enough_snow(arr, clear, water, snow, snow_thresh)


def test_filter_median_green():
    arr = np.arange(10)
    ans = np.array([True, True, True, True, True,
                    True, True, True, False, False])

    assert np.array_equal(ans, filter_median_green(arr, 3))


def test_duplicate_values():
    arr = np.array([1, 2, 2, 3, 4, 5, 5])
    ans = np.array([True, True, False, True, True,
                    True, False], dtype=bool)

    assert np.array_equal(ans, mask_duplicate_values(arr))
