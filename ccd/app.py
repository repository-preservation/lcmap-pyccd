""" Main bootstrap and configuration module for pyccd.  Any module that
requires configuration or services should import app and obtain the
configuration or service from here.

app.py enables a very basic but sufficient form of loose coupling
by setting names of services & configuration once, then allowing other modules
that require these services/information to obtain them by name rather than
directly importing or instantiating.

Module level constructs are only evaluated once in a Python application's
lifecycle, usually at the time of first import. This pattern is borrowed
from Flask.
"""
import logging, sys, yaml, os, hashlib

from cachetools import LRUCache


# Simplify parameter setting and make it easier for adjustment
class Defaults(dict):
    def __init__(self, config_path='parameters.yaml'):
        with open(config_path, 'r') as f:
            super(Defaults, self).__init__(yaml.load(f.read()))

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError('No such attribute: ' + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError('No such attribute: ' + name)


# Don't need to be going down this rabbit hole just yet
# mainly here as reference
def numpy_hashkey(array):
    return hashlib.sha1(array).hexdigest()

# Configuration/parameter defaults
defaults = Defaults(os.path.join(os.path.dirname(__file__), 'parameters.yaml'))


# Logging system
#
# To use the logging from any module:
# import app
# logger = app.logging.getLogger(__name__)
#
# To alter where log messages go or how they are represented,
# configure the logging system below.
# iso8601 date format
__format = '%(asctime)s %(module)-10s::%(funcName)-20s - [%(lineno)-3d]%(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format=__format,
                    datefmt='%Y-%m-%d %H:%M:%S')


# Configure caching
cache = LRUCache(maxsize=2000)

# This is a string.fully.qualified.reference to the fitter function.
# Cannot import and supply the function directly or we'll get a
# circular dependency
FITTER_FN = 'ccd.models.lasso.fitted_model'
