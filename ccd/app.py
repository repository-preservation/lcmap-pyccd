""" Main bootstrap and configuration module for pyccd.  Any module that
requires configuration or services should import app and obtain the
configuration or service from here.

app.py enables a very basic but sufficient form of loose coupling
by setting names of services & configuration once and allowing other modules
that require these services/information to obtain them by name rather than
directly importing or instantiating.

Module level constructs are only evaluated once in a Python application's
lifecycle, usually at the time of first import. This pattern is borrowed
from Flask.
"""
import logging
import sys
from cachetools import LRUCache
from scipy.stats import chi2


############################
# Logging system
############################
# to use the logging from any module:
# import app
# logger = app.logging.getLogger(__name__)
#
# To alter where log messages go or how they are represented,
# configure the
# logging system below.
# iso8601 date format
__format = '%(asctime)s %(module)s::%(funcName)-20s - %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format=__format,
                    datefmt='%Y-%m-%d %H:%M:%S')


# configure caching
cache = LRUCache(maxsize=2000)

############################
# Global configuration items
############################
MINIMUM_CLEAR_OBSERVATION_COUNT = 12

CONSECUTIVE_OBSERVATIONS_COUNT = 6

# 2 for tri-modal; 2 for bi-modal; 2 for seasonality; 2 for linear
COEFFICIENT_CATEGORIES = {'min': 4, 'mid': 6, 'max': 8}

# number of clear observation / number of coefficients
CLEAR_OBSERVATION_THRESHOLD = 3

CLEAR_PCT_THREHOLD = 0.25

SNOW_PCT_THRESHOLD = 0.75

CHANGE_PROBABILITY = 1

# Representative values in the QA band
QA_FILL = 255

QA_CLEAR = 0

QA_WATER = 1

QA_SNOW = 3

MEOW_SIZE = 16

PEEK_SIZE = 3

T_CONST = 4.89

STABILITY_THRESHOLD = 200.0

DETECTION_BANDS = range(2, 7)

QA_BAND = 8

# Tmasking threshold
TMASK_THRESHOLD = chi2.ppf(0.999999, len(DETECTION_BANDS))

# Change detection threshold
CHANGE_THRESHOLD = chi2.ppf(0.99, len(DETECTION_BANDS))

# This is a string.fully.qualified.reference to the fitter function.
# Cannot import and supply the function directly or we'll get a
# circular dependency
FITTER_FN = 'ccd.models.lasso.fitted_model'
