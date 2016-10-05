import shared

def test_basic_loading_sample():
    data = shared.read_data("test/resources/sample_1.csv")
    assert data.shape == (69,9)


def test_filtering_using_no_data():
    data = shared.read_data("test/resources/sample_1.csv")
    masker = data[:,1] != 20000
    assert data[masker,:].shape == (66,9)
