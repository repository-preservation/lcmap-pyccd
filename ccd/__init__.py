import time
import logging

from ccd.procedures import fit_procedure as __determine_fit_procedure, standard_procedure, \
    insufficient_clear_procedure, permanent_snow_procedure
import numpy as np
from ccd import app, math_utils, qa
import importlib
from .version import __version__
from .version import __algorithm__ as algorithm
from .version import __name

log = logging.getLogger(__name)


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
        log.debug(e)
        return None


def __attach_metadata(procedure_results, procedure):
    """
    Attach some information on the algorithm version, what procedure was used,
    and which inputs were used

    Returns:
        A dict representing the change detection results

    {algorithm: 'pyccd:x.x.x',
     processing_mask: (bool, bool, ...),
     procedure: string,
     change_models: [
         {start_day: int,
          end_day: int,
          break_day: int,
          observation_count: int,
          change_probability: float,
          curve_qa: int,
          blue:      {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          green:    {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          red:     {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          nir:      {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          swir1:    {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          swir2:    {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          thermal:  {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float}}
                    ]
    }
    """
    change_models, processing_mask = procedure_results

    return {'algorithm': algorithm,
            'processing_mask': processing_mask,
            'procedure': procedure,
            'change_models': change_models}


def __split_dates_spectra(matrix):
    """ Slice the dates and spectra from the matrix and return """
    return matrix[0], matrix[1:7]


def __sort_dates(dates):
    """ Sort the values chronologically """
    return np.argsort(dates)


def __check_inputs(dates, quality, spectra):
    """
    Make sure the inputs are of the correct relative size to each-other.
    
    Args:
        dates: 1-d ndarray
        quality: 1-d ndarray
        spectra: 2-d ndarray
    """
    # Make sure we only have one dimension
    assert dates.ndim == 1
    # Make sure quality is the same
    assert dates.shape == quality.shape
    # Make sure there is spectral data for each date
    assert dates.shape[0] == spectra.shape[1]


def detect(dates, blues, greens, reds, nirs,
           swir1s, swir2s, thermals, quality,
           params=None):
    """Entry point call to detect change

    No filtering up-front as different procedures may do things
    differently

    Args:
        dates:    1d-array or list of ordinal date values
        blues:    1d-array or list of blue band values
        greens:   1d-array or list of green band values
        reds:     1d-array or list of red band values
        nirs:     1d-array or list of nir band values
        swir1s:   1d-array or list of swir1 band values
        swir2s:   1d-array or list of swir2 band values
        thermals: 1d-array or list of thermal band values
        quality:  1d-array or list of qa band values
        params: python dictionary to change module wide processing
            parameters

    Returns:
        Tuple of ccd.detections namedtuples
    """
    t1 = time.time()

    proc_params = app.get_default_params()

    if params:
        proc_params.update(params)

    dates = np.asarray(dates)
    quality = np.asarray(quality)

    spectra = np.stack((blues, greens,
                        reds, nirs, swir1s,
                        swir2s, thermals))

    __check_inputs(dates, quality, spectra)

    indices = __sort_dates(dates)
    dates = dates[indices]
    spectra = spectra[:, indices]
    quality = quality[indices]

    # load the fitter_fn
    fitter_fn = attr_from_str(proc_params.FITTER_FN)

    if proc_params.QA_BITPACKED is True:
        quality = qa.unpackqa(quality, proc_params)

    # Determine which procedure to use for the detection
    _proc = __determine_fit_procedure(quality,
                                          proc_params.QA_CLEAR,
                                          proc_params.QA_WATER,
                                          proc_params.QA_FILL,
                                          proc_params.QA_SNOW,
                                          proc_params.CLEAR_PCT_THRESHOLD,
                                          proc_params.SNOW_PCT_THRESHOLD)

    if _proc.decode() == "standard_procedure":
        results = standard_procedure(dates, spectra, quality,
                                     proc_params.MEOW_SIZE,
                                     proc_params.PEEK_SIZE,
                                     proc_params.THERMAL_IDX,
                                     proc_params.CURVE_QA['START'],
                                     proc_params.CURVE_QA['END'],
                                     proc_params.QA_CLEAR,
                                     proc_params.QA_WATER,
                                     proc_params.DAY_DELTA,
                                     proc_params.DETECTION_BANDS,
                                     proc_params.TMASK_BANDS,
                                     proc_params.CHANGE_THRESHOLD,
                                     proc_params.T_CONST,
                                     proc_params.AVG_DAYS_YR,
                                     proc_params.LASSO_MAX_ITER,
                                     proc_params.OUTLIER_THRESHOLD,
                                     proc_params.COEFFICIENT_MIN,
                                     proc_params.COEFFICIENT_MID,
                                     proc_params.COEFFICIENT_MAX,
                                     proc_params.NUM_OBS_FACTOR)
    elif _proc.decode() == "permanent_snow_procedure":
        results = permanent_snow_procedure(dates,
                                           spectra,
                                           quality,
                                           proc_params.MEOW_SIZE,
                                           proc_params.CURVE_QA['PERSIST_SNOW'],
                                           proc_params.AVG_DAYS_YR,
                                           proc_params.LASSO_MAX_ITER,
                                           proc_params.COEFFICIENT_MIN,
                                           proc_params.THERMAL_IDX,
                                           proc_params.QA_CLEAR,
                                           proc_params.QA_WATER,
                                           proc_params.QA_SNOW)
    elif _proc.decode() == "insufficient_clear_procedure":
        results = insufficient_clear_procedure(dates,
                                               spectra,
                                               quality,
                                               proc_params.MEOW_SIZE,
                                               proc_params.CURVE_QA['INSUF_CLEAR'],
                                               proc_params.AVG_DAYS_YR,
                                               proc_params.LASSO_MAX_ITER,
                                               proc_params.COEFFICIENT_MIN,
                                               proc_params.GREEN_IDX,
                                               proc_params.MEDIAN_GREEN_FILTER,
                                               proc_params.THERMAL_IDX,
                                               proc_params.QA_CLEAR,
                                               proc_params.QA_WATER)
    else:
        raise Exception("its all gone horribly wrong")

    log.debug('Total time for algorithm: %s', time.time() - t1)
    # call detect and return results as the detections namedtuple
    return __attach_metadata(results, _proc)
