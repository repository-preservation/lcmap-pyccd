from ccd.procedures import determine_fit_procedure as __determine_fit_procedure
from ccd.qa import preprocess as __preprocess
import numpy as np
from ccd import app
import importlib
from .version import __version__
from .version import __algorithm__
from .version import __name

logger = app.logging.getLogger(__name)
config = app.config


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
    """Transforms results of change.detect to the detections dict.

    Args: A tuple as returned from change.detect
            (start_day, end_day, models, errors_, magnitudes_)

    Returns: A dict representing a change detection

        {algorithm:'pyccd:x.x.x',
         start_day:int,
         end_day:int, observation_count:int,
         red:      {magnitudes:float,
                    rmse:float,
                    coefficients:(float, float, ...),
                    intercept:float},
         green:    {magnitudes:float,
                    rmse:float,
                    coefficients:(float, float, ...),
                    intercept:float},
         blue:     {magnitudes:float,
                    rmse:float,
                    coefficients:(float, float, ...),
                    intercept:float},
         nir:     {magnitudes:float,
                    rmse:float,
                    coefficients:(float, float, ...),
                    intercept:float},
         swir1:   {magnitudes:float,
                    rmse:float,
                    coefficients:(float, float, ...),
                    intercept:float},
         swir2:    {magnitudes:float,
                    rmse:float,
                    coefficients:(float, float, ...),
                    intercept:float},
        }
    """
    spectra = ((0, 'red'), (1, 'green'), (2, 'blue'), (3, 'nir'),
               (4, 'swir1'), (5, 'swir2'))

    # get the start and end time for each detection period
    detection = {'algorithm': __algorithm__,
                 'start_day': int(change_tuple[0]),
                 'end_day': int(change_tuple[1]),
                 'observation_count': None,  # dummy value for now
                 'category': None}           # dummy value for now

    # gather the results for each spectra
    for ix, name in spectra:
        model, error, mags = change_tuple[2], change_tuple[3], change_tuple[4]
        _band = {'magnitude': float(mags[ix]),
                 'rmse': float(error[ix]),
                 'coefficients': tuple([float(x) for x in model[ix].coef_]),
                 'intercept': float(model[ix].intercept_)}

        # assign _band to the subdict
        detection[name] = _band

    # build the namedtuple from the dict and return
    return detection


def __as_detections(detect_tuple):
    """Transforms results of change.detect to the detections namedtuple.

    Args: A tuple of dicts as returned from change.detect
        (
            (start_day, end_day, models, errors_, magnitudes_),
            (start_day, end_day, models, errors_, magnitudes_),
            (start_day, end_day, models, errors_, magnitudes_)
        )

    Returns: A tuple of dicts representing change detections
        (
            {},{},{}}
        )
    """
    # iterate over each detection, build the result and return as tuple of
    # dicts
    return tuple([__result_to_detection(t) for t in detect_tuple])


def __split_dates_spectra(matrix):
    """ Slice the dates and spectra from the matrix and return """
    return matrix[0], matrix[1:7]


def detect(dates, reds, greens, blues, nirs,
           swir1s, swir2s, thermals, quality):
    """Entry point call to detect change

    Args:
        dates:    1d-array or list of ordinal date values
        reds:     1d-array or list of red band values
        greens:   1d-array or list of green band values
        blues:    1d-array or list of blue band values
        nirs:     1d-array or list of nir band values
        swir1s:   1d-array or list of swir1 band values
        swir2s:   1d-array or list of swir2 band values
        thermals: 1d-array or list of thermal band values
        quality:  1d-array or list of qa band values

    Returns:
        Tuple of ccd.detections namedtuples
    """
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)

    spectra = np.stack((reds, greens,
                        blues, nirs, swir1s,
                        swir2s, thermals, quality))

    # load the fitter_fn from app.FITTER_FN
    fitter_fn = attr_from_str(config.FITTER_FN)

    # Determine which procedure to use for the detection
    procedure = __determine_fit_procedure(quality)

    # call detect and return results as the detections namedtuple
    return __as_detections(procedure(dates, spectra, fitter_fn,
                                     config.MEOW_SIZE, config.PEEK_SIZE))
