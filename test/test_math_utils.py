from ccd.math_utils import *


def test_euclidean_norm():
    arr = np.arange(5)

    ans = 30.0 ** .5

    assert ans == euclidean_norm(arr)



