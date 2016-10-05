import shared
import pytest
from ccd.filter import *


def test_basic_loading_sample():
    data = shared.read_data("test/resources/sample_1.csv")
    print(data.shape)
    assert data.shape == (69,9), "Sample data an unexpected shape, other tests may fail."


def test_count_snow():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[:,8]
    assert count_snow(qa) == 38


def test_count_clear_or_water():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[:,8]
    assert count_clear_or_water(qa) == 24


def test_count_total():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[:,8]
    assert count_total(qa) == 69


def test_ratio_snow():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[:,8]
    ratio = ratio_snow(qa)
    assert pytest.approx(ratio, 0.61)


def test_ratio_clear():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[:,8]
    ratio = ratio_clear(qa)
    assert pytest.approx(ratio, 0.347)
