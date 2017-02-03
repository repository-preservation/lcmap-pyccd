""" Module specifically to hold algorithm version information.  The reason this
exists is the version information is needed in both setup.py for install and
also in ccd/__init__.py when generating results.  If these values were
defined in ccd/__init__.py then install would fail because there are other
dependencies imported in ccd/__init__.py that are not present until after
install. Do not import anything into this module."""
__version__ = '1.0.3.b1'
__name = 'lcmap-pyccd'
__algorithm__ = ':'.join([__name, __version__])
