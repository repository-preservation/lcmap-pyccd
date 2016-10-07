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

logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s %(module)s::%(funcName)-20s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# configure caching
cache = LRUCache(maxsize=2000)

############################
# Global configuration items
############################
MINIMUM_CLEAR_OBSERVATION_COUNT = 12

# 2 for tri-modal; 2 for bi-modal; 2 for seasonality; 2 for linear
COEFFICIENT_CATEGORIES = {'min': 4, 'mid': 6, 'max': 8}

# number of clear observation / number of coefficients
CLEAR_OBSERVATION_THRESHOLD = 3

CLEAR_OBSERVATION_PCT = 0.25

PERMANENT_SNOW_THRESHOLD = 0.75

CHANGE_PROBABILITY = 1

FILL_VALUE = 255

MEOW_SIZE = 16

PEEK_SIZE = 3

T_CONST = 4.89

# This is a string.fully.qualified.reference to the fitter function.
# Cannot import and supply the function directly or we'll get a
# circular dependency
FITTER_FN = 'ccd.models.lasso.fitted_model'
