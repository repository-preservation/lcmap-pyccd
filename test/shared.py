import numpy as np
import datetime
import aniso8601


def two_change_data():
    """ Generate sample data that has two changes in it.  The qa data is not
    correct at all so this data cannot be used during filtering """
    times = acquisition_delta('R100/2000-01-01/P1M')
    reds = np.hstack((sinusoid(times[0:50])+10, sinusoid(times[50:100])+50))
    greens = sinusoid(times)
    blues = sinusoid(times)
    nirs = sinusoid(times)
    swir1s = sinusoid(times)
    swir2s = sinusoid(times)
    thermals = sinusoid(times)
    qas = sinusoid(times)
    return np.array([times, reds, greens, blues, nirs,
                     swir1s, swir2s, thermals, qas])


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
        generator producing delta in days from 1970-01-01

    Example:
        gen_acquisition_delta('R90/P16D/2000-01-01')
    """
    epoch = datetime.datetime.utcfromtimestamp(0).date()
    dates = gen_acquisition_dates(interval)
    yield [(date-epoch).days for date in dates]


def acquisition_delta(interval):
    """List of delta in days for an interval

    Args:
        interval: ISO8601 date range

    Returns:
        list of deltas in days from 1970-01-01

    Example:
        acquisition_delta('R90/P16D/2000-01-01')
    """
    return list(*gen_acquisition_delta(interval))


def sinusoid(times, frequency=1, amplitude=0.1, seed=42):
    """Produce a sinusoidal wave for testing data"""
    np.random.seed(seed)
    xs = times
    ys = np.array([np.sin(2*np.pi*x/365.2)*amplitude for x in xs])
    return np.array(list([y for y in ys]))
