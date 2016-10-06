from collections import namedtuple
from ccd.change import detect as __detect
from ccd.filter import preprocess as __preprocess
import numpy as np
from ccd import app
import importlib

logger = app.logging.getLogger('ccd')

band = namedtuple("Band", ['magnitude', 'rmse', 'coefficients', 'intercept'])

detections = namedtuple("Detections", ['start_day', 'end_day',
                                       'observation_count',
                                       'red', 'green', 'blue',
                                       'nir', 'swir1', 'swir2',
                                       'category'])


def attr_from_str(value):
    """Returns a reference to the full qualified function, attribute or class.

    Args:
        value = Fully qualified path (e.g. 'ccd.models.lasso.fitted_model')

    Returns:
        A reference to the target attribute (e.g. fitted_model)
    """
    module, target = value.rsplit('.', 1)
    try:
        obj = importlib.import_module(module)
        return getattr(obj, target)
    except (ImportError, AttributeError) as e:
        logger.debug(e)
        return None


def __result_to_detection(change_tuple):
    """Transforms results of change.detect to the detections namedtuple.

    Args: A tuple as returned from change.detect
            (start_day, end_day, models, errors_, magnitudes_)

    Returns: A namedtuple representing a change detection

        (start_day=int, end_day=int, observation_count=int,
         red =     (magnitudes=float,
                    rmse=float,
                    coefficients=(float, float, ...),
                    intercept=float),
         green =   (magnitudes=float,
                    rmse=float,
                    coefficients=(float, float, ...),
                    intercept=float),
         blue =    (magnitudes=float,
                    rmse=float,
                    coefficients=(float, float, ...),
                    intercept=float),
         nir =     (magnitudes=float,
                    rmse=float,
                    coefficients=(float, float, ...),
                    intercept=float),
         swir1 =   (magnitudes=float,
                    rmse=float,
                    coefficients=(float, float, ...),
                    intercept=float),
         swir2 =   (magnitudes=float,
                    rmse=float,
                    coefficients=(float, float, ...),
                    intercept=float),
        )
    """
    spectra = ((0, 'red'), (1, 'green'), (2, 'blue'), (3, 'nir'),
               (4, 'swir1'), (5, 'swir2'))

    # get the start and end time for each detection period
    detection = {'start_day': change_tuple[0],
                 'end_day': change_tuple[1],
                 'observation_count': None,  # dummy value for now
                 'category': None}           # dummy value for now

    # gather the results for each spectra
    for ix, name in spectra:
        model, error, mags = change_tuple[2], change_tuple[3], change_tuple[4]

        # build the inner band namedtuple and add to tmp detection dict()
        _band = band(mags[ix],
                     error[ix],
                     model[ix].coef_,
                     model[ix].intercept_)

        # assign _band to the tmp dict under key as specified in spectra tuple
        detection[name] = _band

    # build the namedtuple from the dict and return
    return detections(**detection)


def __as_detections(detect_tuple):
    """Transforms results of change.detect to the detections namedtuple.

    Args: A tuple of tuples as returned from change.detect
        (
            (start_day, end_day, models, errors_, magnitudes_),
            (start_day, end_day, models, errors_, magnitudes_),
            (start_day, end_day, models, errors_, magnitudes_)
        )

    Returns: A tuple of namedtuples representing change detections
        (
            ccd.detections,
            ccd.detections,
            ccd.detections,
        )
    """
    # iterate over each detection, build the result and return as tuple of
    # namedtuples
    return tuple([__result_to_detection(t) for t in detect_tuple])


def __split_dates_spectra(matrix):
    """ Slice the dates and spectra from the matrix and return """
    return matrix[0], matrix[1:7]


def detect(dates, reds, greens, blues, nirs,
           swir1s, swir2s, thermals, qas, preprocess=True):
    """Entry point call to detect change

    Args:
        dates:    numpy array of ordinal date values
        reds:     numpy array of red band values
        greens:   numpy array of green band values
        blues:    numpy array of blue band values
        nirs:     numpy array of nir band values
        swir1s:   numpy array of swir1 band values
        swir2s:   numpy array of swir2 band values
        thermals: numpy array of thermal band values
        qas:      numpy array of qa band values

    Returns:
        Tuple of ccd.detections namedtuples
    """

    __matrix = np.array([dates, reds, greens,
                         blues, nirs, swir1s,
                         swir2s, thermals, qas])

    # get the spectra separately so we can call detect
    if preprocess is True:
        __dates, __spectra = __split_dates_spectra(__preprocess(__matrix))
    else:
        __dates, __spectra = __split_dates_spectra(__matrix)

    # load the fitter_fn from app.FITTER_FN
    __fitter_fn = attr_from_str(app.FITTER_FN)

    # call detect and return results as the detections namedtuple
    return __as_detections(__detect(__dates, __spectra, __fitter_fn,
                                    app.MEOW_SIZE, app.PEEK_SIZE))
