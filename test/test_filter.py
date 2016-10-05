import shared

def test_basic_loading_sample():
    data = shared.read_data("resources/sample_1.csv")
    assert data.shape == (22,8)


def test_filtering_using_no_data():
    data = shared.read_data("resources/sample_1.csv")
    masker = data[:,1] != 20000
    assert data[masker,:].shape == (17,8)


def test_filtering_cloudy_data_using_masks():
    data = shared.read_data("resources/sample_1.csv")
    assert False, "TODO"


def test_permanent_snow_detection_using_masks():
    data = shared.read_data("resources/sample_1.csv")
    assert False, "TODO"
