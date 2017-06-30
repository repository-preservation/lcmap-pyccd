import aniso8601
import base64
from datetime import datetime
import logging
import math
import numpy as np
import requests
import xarray as xr

log = logging.getLogger(__name__)

# sample1 = read_data("test/resources/sample_1.csv")
# sample2 = read_data("test/resources/sample_2.csv")
# persistent_snow = read_data("test/resources/sample_WA_grid08_row9_col2267_persistent_snow.csv")
# standard_procedure = read_data("test/resources/sample_WA_grid08_row999_col1_normal.csv")
# fmask_fail = read_data("test/resources/sample_WA_grid08_row12_col2265_fmask_fail.csv")


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
    epoch = datetime.utcfromtimestamp(0).date()
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


def sample_sinusoid(time_range, bands=7):
    """Produce N-bands of data with a sample for each moment in time range"""
    times = np.array(acquisition_delta(time_range))
    observations = np.array([sinusoid(times) for _ in range(bands)])
    return times, observations


def line(times, value=300):
    return np.full(len(times), value, dtype=np.int16)


def sample_line(time_range, bands=7):
    """Produce N-bands of data with a sample for each moment in time range"""
    times = np.array(acquisition_delta(time_range))
    observations = np.array([line(times) for _ in range(bands)])
    return times, observations


def near(point, interval, offset):
    """ Calculate the nearest points along the x and y axis """
    return ((math.floor((point - offset) / interval)) * interval) + offset


def point_to_chip(x, y, x_interval, y_interval, x_offset, y_offset):
    """ Transform coordinates into the identifying coordinates of the containing chip """
    return near(x, x_interval, x_offset), near(y, y_interval, y_offset)


def snap(x, y, chip_spec={'shift_y': -195.0, 'shift_x': 2415.0, 'chip_x': 3000, 'chip_y': -3000}):
    """ Identify the chip containing the provided coordinate """
    chip = point_to_chip(x, y, chip_spec['chip_x'], chip_spec['chip_y'], chip_spec['shift_x'], chip_spec['shift_y'])
    return int(chip[0]), int(chip[1])


def get_request(url, params=None):
    """ Return json response for a give url """
    return requests.get(url, params=params).json()


def spectral_map(specs_url):
    """ Return a dict of sensor bands keyed to their respective spectrum """
    _spec_map = dict()
    _map = {'blue': ('sr', 'blue'), 'green': ('sr', 'green'), 'red': ('sr', 'red'), 'nir': ('sr', 'nir'),
            'swir1': ('sr', 'swir1'), 'swir2': ('sr', 'swir2'), 'thermal': ('bt', 'thermal -BTB11'),
            'cfmask': 'pixelqa'}

    try:
        for spectra in _map:
            if isinstance(_map[spectra], str):
                _tags = ["tags:" + _map[spectra]]
            else:
                _tags = ["tags:"+i for i in _map[spectra]]
            _qs = " AND ".join(_tags)
            url = "{specurl}?q=({tags})".format(specurl=specs_url, tags=_qs)
            _start = datetime.now()
            resp = get_request(url)
            dur = datetime.now() - _start
            print("request for {} took {} seconds to fulfill".format(url, dur.total_seconds()))
            # value needs to be a list, make it unique using set()
            _spec_map[spectra] = list(set([i['ubid'] for i in resp]))
        _start_whole = datetime.now()
        _spec_whole = get_request(specs_url)
        _dur_whole = datetime.now() - _start_whole
        print("request for whole spec response to {}, took {} seconds".format(specs_url, _dur_whole.total_seconds()))
    except Exception as e:
        raise Exception("Problem generating spectral map from api query, specs_url: {}\n message: {}".format(specs_url, e))
    return _spec_map, _spec_whole


def as_numpy_array(chip, specs_map):
    """ Return numpy array of chip data grouped by spectral map """
    NUMPY_TYPES = {
        'UINT8': np.uint8,
        'UINT16': np.uint16,
        'INT8': np.int8,
        'INT16': np.int16
    }
    try:
        spec    = specs_map[chip['ubid']]
        np_type = NUMPY_TYPES[spec['data_type']]
        shape   = specs_map[spec['ubid']]['data_shape']
        buffer  = base64.b64decode(chip['data'])
    except KeyError as e:
        raise Exception("as_numpy_array inputs missing expected keys: {}".format(e))

    return np.frombuffer(buffer, np_type).reshape(*shape)


def landsat_dataset(spectrum, ubid, specs, chips):
    """ Return stack of landsat data for a given ubid, x, y, and time-span """
    # specs may not be unique, deal with it
    uniq_specs = []
    for spec in specs:
        if spec not in uniq_specs:
            uniq_specs.append(spec)

    specs_map = dict([[spec['ubid'], spec] for spec in uniq_specs if spec['ubid'] == ubid])
    rasters = xr.DataArray([as_numpy_array(chip, specs_map) for chip in chips])

    ds = xr.Dataset()
    ds[spectrum] = (('t', 'x', 'y'), rasters)
    ds[spectrum].attrs = {'color': spectrum}
    #ds.coords['t'] = (('t'), pd.to_datetime([t['acquired'] for t in chips]))
    ds.coords['t'] = (('t'), [t['acquired'] for t in chips])
    return ds


def rainbow(x, y, t, specs_url, chips_url, requested_ubids):
    """ Return all the landsat data, organized by spectra for a given x, y, and time-span """
    spec_map, spec_whole = spectral_map(specs_url)
    ds = xr.Dataset()
    for (spectrum, ubids) in spec_map.items():
        for ubid in ubids:
            if ubid in requested_ubids:
                params = {'ubid': ubid, 'x': x, 'y': y, 'acquired': t}
                _chip_start = datetime.now()
                chips_resp = get_request(chips_url, params=params)
                _chip_dur = datetime.now() - _chip_start
                print("chip request for ubid, x, y, acquired: {}, {}, {}, {} "
                      "took: {} seconds".format(ubid, x, y, t, _chip_dur.total_seconds()))
                if chips_resp:
                    band = landsat_dataset(spectrum, ubid, spec_whole, chips_resp)
                    if band:
                        # combine_first instead of merge, for locations where data is missing for some bands
                        ds = ds.combine_first(band)
    return ds.fillna(0)
