"""
Tests for the basic masking and filtering operations
"""
import numpy as np

import pytest
from ccd.qa import *
from ccd.app import defaults


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
    ans = np.array([True, True, False, False, False])

    assert np.array_equal(ans, mask_clear_or_water(arr))


def test_count_clear_or_water():
    arr = np.arange(5)
    ans = 2

    assert ans == count_clear_or_water(arr)


def test_count_fill():
    arr = np.arange(5)
    arr[-1] = 255
    ans = 1

    assert ans == count_fill(arr)


def test_count_snow():
    arr = np.arange(5)
    ans = 1

    assert ans == count_snow(arr)


def test_count_total():
    arr = np.arange(5)
    arr[-1] = 255
    ans = 4

    assert ans == count_total(arr)


def test_ratio_clear():
    arr = np.arange(5)
    arr[-1] = 255
    ans = 0.5

    assert ans == ratio_clear(arr)


def test_ratio_snow():
    arr = np.arange(5)
    arr[-1] = 255
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

    arr[1:] = 3
    ans = True
    assert ans == enough_snow(arr)


def test_filter_median_green():
    arr = np.arange(10)
    ans = np.array([True, True, True, True, True,
                    True, True, True, False, False])

    assert np.array_equal(ans, filter_median_green(arr, 3))
