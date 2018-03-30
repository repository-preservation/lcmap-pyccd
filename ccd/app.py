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
import yaml, os, hashlib, zipfile


# Simplify parameter setting and make it easier for adjustment
class Parameters(dict):
    def __init__(self, config_path='parameters.yaml'):
        if '.zip' in config_path:
            zp, ym = config_path.split('.zip/')

            with zipfile.ZipFile('{}.zip'.format(zp), 'r') as myzip:
                with myzip.open(ym) as f:
                    conf = f.read()

        else:
            with open(config_path, 'r') as f:
                conf = f.read()

        super(Parameters, self).__init__(yaml.load(conf))

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


# This is a string.fully.qualified.reference to the fitter function.
# Cannot import and supply the function directly or we'll get a
# circular dependency
FITTER_FN = 'ccd.models.lasso.fitted_model'


def get_default_params():
    return Parameters(os.path.join(os.path.dirname(__file__), 'parameters.yaml'))
