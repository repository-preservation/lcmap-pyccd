from ccd.math_utils import *


def test_ensure_ndarray():
    @ensure_ndarray_input(keywords=False)
    def nokeyword_func(arg1):
        return arg1

    @ensure_ndarray_input(keywords=True)
    def keyword_func(arg1, arg2=1):
        return arg1, arg2

    assert isinstance(nokeyword_func(1), np.ndarray)

    a1, a2 = keyword_func(1, arg2=2)
    assert isinstance(a1, np.ndarray)
    assert isinstance(a2, int)
