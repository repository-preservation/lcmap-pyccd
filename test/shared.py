import numpy as np

def read_data(path):
    """Load a sample file containing acquisition days and spectral values.

    The first column is assumed to be the day number, subsequent columns
    correspond to the day number. This improves readability of large datasets.

    Args:
        path: location of CSV containing test data

    Returns:
        A 2D numpy array.
    """
    return np.genfromtxt(path, delimiter=',', dtype=np.int).T
