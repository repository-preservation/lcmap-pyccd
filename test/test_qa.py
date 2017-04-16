"""
Tests for the basic masking and filtering operations
"""
from ccd.qa import *


def test_checkbit():
    packint = 1
    offset = 0

    assert checkbit(packint, offset)

    offset = 1

    assert not checkbit(packint, offset)


def test_qabitval():
    fill = 0
    clear = 1
    water = 2
    shadow = 3
    snow = 4
    cloud = 5

    packint = 3

    assert qabitval(packint, fill=fill, clear=clear, water=water,
                    shadow=shadow, snow=snow, cloud=cloud) == fill


def test_mask_snow():
    arr = np.arange(5)
    ans = np.array([False, False, False, True, False])

    assert np.array_equal(ans, mask_snow(arr, 3))


def test_mask_clear():
    arr = np.arange(5)
    ans = np.array([True, False, False, False, False])

    assert np.array_equal(ans, mask_clear(arr, 0))


def test_mask_water():
    arr = np.arange(5)
    ans = np.array([False, True, False, False, False])

    assert np.array_equal(ans, mask_water(arr, 1))


def test_mask_fill():
    arr = np.arange(5)
    arr[-1] = 255
    ans = np.array([False, False, False, False, True])

    assert np.array_equal(ans, mask_fill(arr, 255))


def test_mask_clear_or_water():
    arr = np.arange(5)
    ans = np.array([False, True, True, False, False])

    assert np.array_equal(ans, mask_clear_or_water(arr))


def test_count_clear_or_water():
    arr = np.arange(5)
    ans = 2

    assert ans == count_clear_or_water(arr)


def test_count_fill():
    arr = np.arange(5)
    ans = 1

    assert ans == count_fill(arr)


def test_count_snow():
    arr = np.arange(5)
    ans = 1

    assert ans == count_snow(arr)


def test_count_total():
    arr = np.arange(5)
    ans = 4

    assert ans == count_total(arr)


def test_ratio_clear():
    arr = np.arange(5)
    ans = 0.5

    assert ans == ratio_clear(arr)


def test_ratio_snow():
    arr = np.arange(5)
    ans = 1/3.01

    assert ans == ratio_snow(arr)


def test_enough_clear():
    arr = np.arange(5)
    ans = True

    assert ans == enough_clear(arr)

    arr[1:] = 4
    ans = False
    assert ans == enough_clear(arr)


def test_enough_snow():
    arr = np.arange(5)
    ans = False

    assert ans == enough_snow(arr)

    arr[1:] = 4
    ans = True
    assert ans == enough_snow(arr)


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
