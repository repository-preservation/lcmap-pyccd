import shared
import pytest
from ccd.filter import *

#
# Sample 1 (test/resources/sample_1.csva)
#

def test_basic_loading_sample():
    data = shared.read_data("test/resources/sample_1.csv")
    print(data.shape)
    assert data.shape == (9,69), "Sample data an unexpected shape, other tests may fail."


def test_count_snow():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[8,:]
    assert count_snow(qa) == 38


def test_count_clear_or_water():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[8,:]
    assert count_clear_or_water(qa) == 24


def test_count_total():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[8,:]
    assert count_total(qa) == 69


def test_ratio_snow():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[8,:]
    ratio = ratio_snow(qa)
    assert ratio == pytest.approx(0.61, rel=1e-2)


def test_ratio_clear():
    data = shared.read_data("test/resources/sample_1.csv")
    qa = data[8,:]
    ratio = ratio_clear(qa)
    assert ratio == pytest.approx(0.347, rel=1e-2)

#
# Sample 2 (test/resources/sample_2.csv)
#

def test_basic_loading_sample():
    data = shared.read_data("test/resources/sample_2.csv")
    print(data.shape)
    assert data.shape == (9,724), "Sample data an unexpected shape, other tests may fail."


def test_sample_2_count_snow():
    data = shared.read_data("test/resources/sample_2.csv")
    qa = data[8,:]
    assert count_snow(qa) == 214, count_snow(qa)


def test_sample_2_count_clear_or_water():
    data = shared.read_data("test/resources/sample_2.csv")
    qa = data[8,:]
    assert count_clear_or_water(qa) == 480, count_clear_or_water(qa)


def test_sample_2_count_total():
    data = shared.read_data("test/resources/sample_2.csv")
    qa = data[8,:]
    assert count_total(qa) == 724, count_total(qa)


def test_sample_2_ratio_snow():
    data = shared.read_data("test/resources/sample_2.csv")
    qa = data[8,:]
    ratio = ratio_snow(qa)
    # print(ratio)
    assert ratio == pytest.approx(0.308, rel=1e-2)


def test_sample_2_ratio_clear():
    data = shared.read_data("test/resources/sample_2.csv")
    qa = data[8,:]
    ratio = ratio_clear(qa)
    assert ratio == pytest.approx(0.662, rel=1e-2)

def test_sample_2_exercise_temperature_index():
    # This sample data does not have any thermal values that
    # are outside the threshold... this test still serves
    # a purpose though.
    data = shared.read_data("test/resources/sample_2.csv")
    index = temperature_index(data)
    assert data[:,index].shape == (9,724)

def test_sample_2_preprocessing():
    data = shared.read_data("test/resources/sample_2.csv")
    meow = preprocess(data)
    assert meow.shape == (9,477), meow.shape
