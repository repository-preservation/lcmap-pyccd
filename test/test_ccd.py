from itertools import cycle, islice
import aniso8601
import datetime
import numpy as np

<<<<<<< HEAD
=======
from ccd.models import lasso
from itertools import cycle, islice


# Test data generators for change detection.
>>>>>>> 0410a69a941268c57f979bf61daf24f551ba9e20

def gen_acquisition_dates(interval):
    """Generate acquisition dates for an ISO8601 interval.

    Args:
        interval: An ISO8610 interval.

    Returns:
        generator producing datetime.date or datetime.datetime
        depending on the resolution of the interval.

    Example:
        gen_acquisition_dates('R10/2015-01-01/P16D')
    """
    dates = aniso8601.parse_repeating_interval(interval)
    return dates


def gen_acquisition_delta(interval):
    """Generate delta in days for an acquisition day since 1970-01-01.

    Args:
        interval: ISO8601 date range

    Returns:

    Example:

    """
    epoch = datetime.datetime.utcfromtimestamp(0).date()
    dates = gen_acquisition_dates(interval)
    yield [(date-epoch).days for date in dates]


<<<<<<< HEAD
=======
def read_csv_sample(path):
    """Load a sample file containing acquisition days and spectral values"""
    return np.genfromtxt('test/resources/sample_1.csv', delimiter=',')


>>>>>>> 0410a69a941268c57f979bf61daf24f551ba9e20
def acquisition_delta(interval):
    """ List of delta in days for an interval """
    return list(*gen_acquisition_delta(interval))


def repeated_values(samples, seed=42):
    np.random.seed(seed)
    sine = np.array(list(islice(cycle([0, 1, 0, -1]), None, samples)))
    noise = np.array(np.random.random(samples))
    return sine+noise


def test_not_enough_observations():
    acquired = acquisition_delta('R15/P16D/2000-01-01')
    observes = repeated_values(15)
    assert len(acquired) == len(observes) == 15


def test_enough_observations():
    acquired = acquisition_delta('R16/P16D/2000-01-01')
    observes = repeated_values(16)
    assert len(acquired) == len(observes) == 16


def test_two_changes_during_time():
    acquired = acquisition_delta('R50/P16D/2000-01-01')
    observes = np.hstack((repeated_values(25) + 10,
                          repeated_values(25) + 50))
    assert len(acquired) == len(observes) == 50


def test_three_changes_during_time():
    acquired = acquisition_delta('R90/P16D/2000-01-01')
    observes = np.hstack((repeated_values(30) + 10,
                          repeated_values(30) + 50,
                          repeated_values(30) + 10))
    assert len(acquired) == len(observes) == 90
