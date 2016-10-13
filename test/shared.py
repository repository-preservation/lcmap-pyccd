import numpy as np
import datetime
import aniso8601
import ccd.app as app


log = app.logging.getLogger(__name__)


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
    xs = np.linspace(0, 2*np.pi, len(times))
    ys = np.array([np.sin(x)*amplitude for x in xs])
    scaled_ys = np.array(ys*100+1000, dtype=np.int16)
    log.debug(scaled_ys)
    return np.array(scaled_ys)


def sample_sinusoid(time_range, bands = 6):
    """Produce N-bands of data with a sample for each moment in time range"""
    times = np.array(acquisition_delta(time_range))
    observations = np.array([sinusoid(times) for _ in range(bands)])
    return times, observations


def line(times, value = 1000):
    return np.full(len(times), value, dtype=np.int16)


def sample_line(time_range, bands = 6):
    """Produce N-bands of data with a sample for each moment in time range"""
    times = np.array(acquisition_delta(time_range))
    observations = np.array([line(times) for _ in range(bands)])
    return times, observations
