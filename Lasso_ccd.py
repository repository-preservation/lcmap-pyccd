import warnings
# from ..utils import ConvergenceWarning

import numpy as np
import scipy.sparse as sp
import numbers
from cachetools import LRUCache
import pandas as pd
from datetime import datetime
from abc import ABCMeta, abstractmethod
import sys

import six
from six.moves import xrange

import pprint
pp = pprint.PrettyPrinter()

from lasso_externals import signature

# mingw_setup_args={'options': {'build_ext': {'compiler': 'mingw32'}}}
# import pyximport; pyximport.install(setup_args=mingw_setup_args)
#sys.path.append('C:\\Program Files\\Anaconda2\\Lib\\site-packages\\sklearn\\linear_model\\')
# import cd_fast

class NotFittedError(Exception):
    pass

def mean_variance_axis(X, axis):
    """Compute mean and variance along an axix on a CSR or CSC matrix
    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
        Input data.
    axis : int (either 0 or 1)
        Axis along which the axis should be computed.
    Returns
    -------
    means : float array with shape (n_features,)
        Feature-wise means
    variances : float array with shape (n_features,)
        Feature-wise variances
    """
    _raise_error_wrong_axis(axis)

    if isinstance(X, sp.csr_matrix):
        if axis == 0:
            return _csr_mean_var_axis0(X)
        else:
            return _csc_mean_var_axis0(X.T)
    elif isinstance(X, sp.csc_matrix):
        if axis == 0:
            return _csc_mean_var_axis0(X)
        else:
            return _csr_mean_var_axis0(X.T)
    else:
        _raise_typeerror(X)

def sparse_center_data(X, y, fit_intercept, normalize=False):
    """
    Compute information needed to center data to have mean zero along
    axis 0. Be aware that X will not be centered since it would break
    the sparsity, but will be normalized if asked so.
    """
    if fit_intercept:
        # we might require not to change the csr matrix sometimes
        # store a copy if normalize is True.
        # Change dtype to float64 since mean_variance_axis accepts
        # it that way.
        if sp.isspmatrix(X) and X.getformat() == 'csr':
            X = sp.csr_matrix(X, copy=normalize, dtype=np.float64)
        else:
            X = sp.csc_matrix(X, copy=normalize, dtype=np.float64)

        X_offset, X_var = mean_variance_axis(X, axis=0)
        if normalize:
            # transform variance to std in-place
            X_var *= X.shape[0]
            X_std = np.sqrt(X_var, X_var)
            del X_var
            X_std[X_std == 0] = 1
            inplace_column_scale(X, 1. / X_std)
        else:
            X_std = np.ones(X.shape[1])
        y_offset = y.mean(axis=0)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1])
        X_std = np.ones(X.shape[1])
        y_offset = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_std

def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
        matrix input will raise an error.  If the input is sparse but not in
        the allowed format, it will be converted to the first listed format.

    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if accept_sparse in [None, False]:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False
    if (isinstance(accept_sparse, (list, tuple))
            and spmatrix.format not in accept_sparse):
        # create new with correct sparse
        spmatrix = spmatrix.asformat(accept_sparse[0])
        changed_format = True

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data)
    return spmatrix
#####################sklearn.utils.validation################################
def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)

def _shape_repr(shape):
    """Return a platform independent reprensentation of an array shape

    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).

    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.

    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.

    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined

def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: "
                         "%s" % str(uniques))

def check_array(array, accept_sparse=None, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is converted to an at least 2nd numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        array = np.array(array, dtype=dtype, order=order, copy=copy)

        if ensure_2d:
            if array.ndim == 1:
                if ensure_min_samples >= 2:
                    raise ValueError("%s expects at least 2 samples provided "
                                     "in a 2 dimensional array-like input"
                                     % estimator_name)
                warnings.warn(
                    "Passing 1d arrays as data is deprecated in 0.17 and will"
                    "raise ValueError in 0.19. Reshape your data either using "
                    "X.reshape(-1, 1) if your data has a single feature or "
                    "X.reshape(1, -1) if it contains a single sample.",
                    DeprecationWarning)
            array = np.atleast_2d(array)
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # make sure we acually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        #warnings.warn(msg, DataConversionWarning)
    return array


def check_X_y(X, y, accept_sparse=None, dtype="numeric", order=None, copy=False,
              force_all_finite=True, ensure_2d=True, allow_nd=False,
              multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X 2d and y 1d.
    Standard input checks are only applied to y. For multi-label y,
    set multi_output=True to allow 2d and sparse y.
    If the dtype of X is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2-d y (array or sparse matrix). If false, y will be
        validated as a vector.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = RegressorMixin.column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y

def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return check_array(X, ['csr', 'csc', 'coo'], dtype=np.float64,
                           copy=copy, force_all_finite=force_all_finite,
                           ensure_2d=False)
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        return X.astype(np.float32 if X.dtype == np.int32 else np.float64)

def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output``=False.
    """
    if sp.issparse(a) or sp.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)

##########################Linear_model.base#################################
def _pre_fit(X, y, Xy, precompute, normalize, fit_intercept, copy):
    """Aux function used at beginning of fit in linear models"""
    n_samples, n_features = X.shape

    if sp.isspmatrix(X):
        precompute = False
        X, y, X_mean, y_mean, X_std = sparse_center_data(
            X, y, fit_intercept, normalize)
    else:
        # copy was done in fit if necessary
        X, y, X_mean, y_mean, X_std = center_data(
            X, y, fit_intercept, normalize, copy=copy)
    if hasattr(precompute, '__array__') and (
            fit_intercept and not np.allclose(X_mean, np.zeros(n_features))
            or normalize and not np.allclose(X_std, np.ones(n_features))):
        warnings.warn("Gram matrix was provided but X was centered"
                      " to fit intercept, "
                      "or X was normalized : recomputing Gram matrix.",
                      UserWarning)
        # recompute Gram
        precompute = 'auto'
        Xy = None

    # precompute if n_samples > n_features
    if isinstance(precompute, six.string_types) and precompute == 'auto':
        precompute = (n_samples > n_features)

    if precompute is True:
        # make sure that the 'precompute' array is contiguous.
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype,
                              order='C')
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, '__array__'):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, '__array__') and Xy is None:
        common_dtype = np.find_common_type([X.dtype, y.dtype], [])
        if y.ndim == 1:
            # Xy is 1d, make sure it is contiguous.
            Xy = np.empty(shape=n_features, dtype=common_dtype, order='C')
            np.dot(X.T, y, out=Xy)
        else:
            # Make sure that Xy is always F contiguous even if X or y are not
            # contiguous: the goal is to make it fast to extract the data for a
            # specific target.
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype,
                          order='F')
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_mean, y_mean, X_std, precompute, Xy

def center_data(X, y, fit_intercept, normalize=False, copy=True,
                sample_weight=None):
    """
    Centers data to have mean zero along axis 0. This is here because
    nearly all linear models will want their data to be centered.

    If sample_weight is not None, then the weighted mean of X and y
    is zero, and not the mean itself
    """
    X = as_float_array(X, copy)
    if fit_intercept:
        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sp.issparse(X):
            X_mean = np.zeros(X.shape[1])
            X_std = np.ones(X.shape[1])
        else:
            X_mean = np.average(X, axis=0, weights=sample_weight)
            X -= X_mean
            if normalize:
                # XXX: currently scaled to variance=n_samples
                X_std = np.sqrt(np.sum(X ** 2, axis=0))
                X_std[X_std == 0] = 1
                X /= X_std
            else:
                X_std = np.ones(X.shape[1])
        y_mean = np.average(y, axis=0, weights=sample_weight)
        y = y - y_mean
    else:
        X_mean = np.zeros(X.shape[1])
        X_std = np.ones(X.shape[1])
        y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
    return X, y, X_mean, y_mean, X_std

def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True,
                eps=1e-3, n_alphas=100, normalize=False, copy_X=True):
    """ Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray, shape (n_samples,)
        Target values

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed.

    l1_ratio : float
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For
        l1_ratio = 1`` it is an L1 penalty.  For ``0 < l1_ratio <
        1``, the penalty is a combination of L1 and L2.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    fit_intercept : boolean, default True
        Whether to fit an intercept or not

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    n_samples = len(y)

    sparse_center = False
    if Xy is None:
        X_sparse = sparse.isspmatrix(X)
        sparse_center = X_sparse and (fit_intercept or normalize)
        X = check_array(X, 'csc',
                        copy=(copy_X and fit_intercept and not X_sparse))
        if not X_sparse:
            # X can be touched inplace thanks to the above line
            X, y, _, _, _ = center_data(X, y, fit_intercept,
                                        normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        if sparse_center:
            # Workaround to find alpha_max for sparse matrices.
            # since we should not destroy the sparsity of such matrices.
            _, _, X_mean, _, X_std = sparse_center_data(X, y, fit_intercept,
                                                        normalize)
            mean_dot = X_mean * np.sum(y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
        if normalize:
            Xy /= X_std[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                       num=n_alphas)[::-1]


def fmax(x, y):
    if x > y:
        return x
    return y


def fsign(f):
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


def enet_coordinate_descent(w, alpha, X, y, max_iter, tol,
                            positive=0):  # beta = 0 for Lasso; not used parameters: beta, rng, random,
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2

    """

    # get the data information into easy vars
    n_features = X.shape[1]

    # compute norms of the columns of X
    norm_cols_X = (X ** 2).sum(axis=0)

    gap = tol + 1.0
    d_w_tol = tol
    n_iter = 0

    R = y - np.dot(X, w)

    tol *= np.dot(y, y)

    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_features):  # Loop over coordinates
            ii = f_iter  ######## random selection doesn't work anymore

            if norm_cols_X[ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value

            if w_ii != 0.0:
                R += w_ii * X[:, ii]

            tmp = (X[:, ii] * R).sum()

            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = (fsign(tmp) * fmax(np.abs(tmp) - alpha, 0)
                         / (norm_cols_X[ii]))

            if w[ii] != 0.0:
                R -= w[ii] * X[:, ii]  # Update residual

            # update the maximum absolute coefficient update
            d_w_ii = np.abs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if np.abs(w[ii]) > w_max:
                w_max = np.abs(w[ii])

        if (w_max == 0.0 or
                        d_w_max / w_max < d_w_tol or
                    n_iter == max_iter - 1):
            # the biggest coordinate update of this iteration was smaller
            # than the tolerance: check the duality gap as ultimate
            # stopping criterion

            XtA = np.dot(X.T, R)

            if positive:
                dual_norm_XtA = np.max(XtA)
            else:
                dual_norm_XtA = np.max(np.abs(XtA))

            R_norm2 = np.dot(R, R)

            if (dual_norm_XtA > alpha):
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            l1_norm = sum(np.abs(w))

            gap += (alpha * l1_norm - const * np.dot(R.T, y))

            if gap < tol:
                # return if we reached desired tolerance
                break
    return w, gap, tol, n_iter + 1


def sparse_enet_coordinate_descent(w, alpha, X_data, X_indices, X_indptr, y, X_mean, max_iter, tol, positive=0):
    """Cython version of the coordinate descent algorithm for Elastic-Net

    We minimize:

        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2

    """

    # get the data information into easy vars
    n_samples = y.shape[0]
    n_features = w.shape[0]

    startptr = X_indptr[0]

    # initial value of the residuals
    R = y.copy()

    norm_cols_X = np.zeros(n_features, dtype=dtype)
    X_T_R = np.zeros(n_features, dtype=dtype)
    XtA = np.zeros(n_features, dtype=dtype)

    R_sum = 0.0
    gap = tol + 1.0
    d_w_tol = tol
    n_iter = 0

    center = (X_mean != 0).any()

    for ii in range(n_features):
        X_mean_ii = X_mean[ii]
        endptr = X_indptr[ii + 1]
        normalize_sum = 0.0
        w_ii = w[ii]

        for jj in range(startptr, endptr):
            normalize_sum += (X_data[jj] - X_mean_ii) ** 2
            R[X_indices[jj]] -= X_data[jj] * w_ii
        norm_cols_X[ii] = normalize_sum + \
                          (n_samples - endptr + startptr) * X_mean_ii ** 2

        if center:
            for jj in range(n_samples):
                R[jj] += X_mean_ii * w_ii
        startptr = endptr

    tol *= np.dot(y, y)

    for n_iter in range(max_iter):

        w_max = 0.0
        d_w_max = 0.0

        for f_iter in range(n_features):  # Loop over coordinates
            ii = f_iter

            if norm_cols_X[ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value
            X_mean_ii = X_mean[ii]

            if w_ii != 0.0:
                R += w_ii * X[:, ii]
                if center:
                    for jj in range(n_samples):
                        R[jj] -= X_mean_ii * w_ii

            tmp = (X[:, ii] * R).sum()

            if center:
                R_sum = 0.0
                for jj in range(n_samples):
                    R_sum += R[jj]
                tmp -= R_sum * X_mean_ii

            if positive and tmp < 0.0:
                w[ii] = 0.0
            else:
                w[ii] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                        / (norm_cols_X[ii])

            if w[ii] != 0.0:
                R -= w[ii] * X[:, ii]  # Update residual

                if center:
                    for jj in range(n_samples):
                        R[jj] += X_mean_ii * w[ii]

            # update the maximum absolute coefficient update
            d_w_ii = fabs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if w[ii] > w_max:
                w_max = w[ii]
        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller than
            # the tolerance: check the duality gap as ultimate stopping
            # criterion

            # sparse X.T / dense R dot product
            if center:
                R_sum = 0.0
                for jj in range(n_samples):
                    R_sum += R[jj]

            for ii in range(n_features):
                X_T_R[ii] = 0.0
                for jj in range(X_indptr[ii], X_indptr[ii + 1]):
                    X_T_R[ii] += X_data[jj] * R[X_indices[jj]]

                if center:
                    X_T_R[ii] -= X_mean[ii] * R_sum
                XtA[ii] = X_T_R[ii]

            if positive:
                dual_norm_XtA = np.max(XtA)
            else:
                dual_norm_XtA = np.max(np.abs(XtA))

            R_norm2 = np.dot(R, R)

            if (dual_norm_XtA > alpha):
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * const ** 2
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            l1_norm = sum(np.abs(w))
            gap += alpha * l1_norm - const * np.dot(R.T, y)

        if gap < tol:
            # return if we reached desired tolerance
            break

    return w, gap, tol, n_iter + 1


def enet_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False, positive=False,
              check_input=True, **params):
    """Compute elastic net path with coordinate descent
    
    The elastic net optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        1 / (2 * n_samples) * ||y - Xw||^2_2 +
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Target values

    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.

    verbose : bool or integer
        Amount of verbosity.

    params : kwargs
        keyword arguments passed to the coordinate descent solver.

    return_n_iter : bool
        whether to return the number of iterations or not.

    positive : bool, default False
        If set to True, forces coefficients to be positive.

    check_input : bool, default True
        Skip input validation checks, including the Gram matrix when provided
        assuming there are handled by the caller when check_input=False.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
        (Is returned when ``return_n_iter`` is set to True).

    Notes
    -----
    See examples/plot_lasso_coordinate_descent_path.py for an example.

    See also
    --------
    MultiTaskElasticNet
    MultiTaskElasticNetCV
    ElasticNet
    ElasticNetCV
    """
    # We expect X and y to be already float64 Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(X, 'csc', dtype=np.float64, order='F', copy=copy_X)
        y = check_array(y, 'csc', dtype=np.float64, order='F', copy=False,
                        ensure_2d=False)
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(Xy, dtype=np.float64, order='C', copy=False,
                             ensure_2d=False)
    n_samples, n_features = X.shape

    # MultiTaskElasticNet does not support sparse matrices
    if sp.isspmatrix(X):
        if 'X_mean' in params:
            # As sparse matrices are not actually centered we need this
            # to be passed to the CD solver.
            X_sparse_scaling = params['X_mean'] / params['X_std']
        else:
            X_sparse_scaling = np.zeros(n_features)

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if check_input:
        X, y, X_mean, y_mean, X_std, precompute, Xy = \
            _pre_fit(X, y, Xy, precompute, normalize=False,
                     fit_intercept=False, copy=False)
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")

    coefs = np.empty((n_features, n_alphas), dtype=np.float64)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1]))
    else:
        coef_ = np.asfortranarray(coef_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        if sp.isspmatrix(X):
            model = sparse_enet_coordinate_descent(
                coef_, l1_reg, X.data, X.indices,
                X.indptr, y, X_sparse_scaling,
                max_iter, tol, positive)
        elif precompute is False:
            model = enet_coordinate_descent(
                coef_, l1_reg, X, y, max_iter, tol, positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like")
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
            warnings.warn('Objective did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations')

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    >>> from sklearn.utils import deprecated
    >>> deprecated() # doctest: +ELLIPSIS
    <sklearn.utils.deprecated object at ...>

    >>> @deprecated()
    ... def some_function(): pass
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=''):
        """
        Parameters
        ----------
        extra: string
          to be added to the deprecation messages

        """
        self.extra = extra

    def __call__(self, obj):
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


class BaseEstimator(object):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, pp.pprint(self.get_params(deep=False),),)
        #return '%s(%s)' % (class_name, pp.pprint(self.get_params(deep=False), offset=len(class_name), ),)


class LinearModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    @deprecated(" and will be removed in 0.19.")
    def decision_function(self, X):
        """Decision function of the linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    def _decision_function(self, X):
        check_is_fitted(self, "coef_")

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return safe_sparse_dot(X, self.coef_.T,
                               dense_output=True) + self.intercept_

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    _center_data = staticmethod(center_data)

    def _set_intercept(self, X_mean, y_mean, X_std):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.


class RegressorMixin(object):
    """Mixin class for all regression estimators in scikit-learn."""
    _estimator_type = "regressor"

    def column_or_1d(self, y, warn=False):
        """ ripped from sklearn """
        shape = np.shape(y)
        if len(shape) == 1:
            return np.ravel(y)
        if len(shape) == 2 and shape[1] == 1:
            #if warn:
            #    warnings.warn("A column-vector y was passed when a 1d array was"
            #                  " expected. Please change the shape of y to "
            #                  "(n_samples, ), for example using ravel().",
            #                  DataConversionWarning, stacklevel=2)
            return np.ravel(y)

        raise ValueError("bad input shape {0}".format(shape))

    def _check_reg_targets(self, y_true, y_pred, multioutput):
        """ ripped from sklearn """
        check_consistent_length(y_true, y_pred)
        y_true = check_array(y_true, ensure_2d=False)
        y_pred = check_array(y_pred, ensure_2d=False)

        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape((-1, 1))

        if y_true.shape[1] != y_pred.shape[1]:
            raise ValueError("y_true and y_pred have different number of output "
                             "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

        n_outputs = y_true.shape[1]
        allowed_multioutput_str = ('raw_values', 'uniform_average',
                                   'variance_weighted')
        if isinstance(multioutput, six.string_types):
            if multioutput not in allowed_multioutput_str:
                raise ValueError("Allowed 'multioutput' string values are {}. "
                                 "You provided multioutput={!r}".format(allowed_multioutput_str, multioutput))
        elif multioutput is not None:
            multioutput = check_array(multioutput, ensure_2d=False)
            if n_outputs == 1:
                raise ValueError("Custom weights are useful only in multi-output cases.")
            elif n_outputs != len(multioutput):
                raise ValueError(("There must be equally many custom weights (%d) as outputs (%d).") % (len(multioutput), n_outputs))

        y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

        return y_type, y_true, y_pred, multioutput

    def r2_score(self, y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
        """ ripped from sklearn """
        y_type, y_true, y_pred, multioutput = self._check_reg_targets(
            y_true, y_pred, multioutput)

        if sample_weight is not None:
            sample_weight = self.column_or_1d(sample_weight)
            weight = sample_weight[:, np.newaxis]
        else:
            weight = 1.

        numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                          dtype=np.float64)
        denominator = (weight * (y_true - np.average(
            y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                              dtype=np.float64)
        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = np.ones([y_true.shape[1]])
        output_scores[valid_score] = 1 - (numerator[valid_score] /
                                          denominator[valid_score])
        # arbitrary set to zero to avoid -inf scores, having a constant
        # y_true is not interesting for scoring a regression anyway
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
        avg_weights = None # CA 7/19/17
        if isinstance(multioutput, six.string_types):
            if multioutput == 'raw_values':
                # return scores individually
                return output_scores
            elif multioutput == 'uniform_average':
                # passing None as weights results is uniform mean
                avg_weights = None
            elif multioutput == 'variance_weighted':
                avg_weights = denominator
                # avoid fail on constant y or one-element arrays
                if not np.any(nonzero_denominator):
                    if not np.any(nonzero_numerator):
                        return 1.0
                    else:
                        return 0.0
        else:
            avg_weights = multioutput

        return np.average(output_scores, weights=avg_weights)

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        #from .metrics import r2_score
        #from sklearn.metrics import r2_score
        return self.r2_score(y, LinearModel().predict(X), sample_weight=sample_weight, multioutput='variance_weighted')


class ElasticNet(LinearModel, RegressorMixin):
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2 +
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * L1 + b * L2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0
        See the notes for the exact mathematical meaning of this
        parameter.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the Lasso object is not advised
        and you should prefer the LinearRegression object.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.
        WARNING : The ``'auto'`` option is deprecated and will
        be removed in 0.18.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Notes
    -----
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    See also
    --------
    SGDRegressor: implements elastic net regression with incremental training.
    SGDClassifier: implements logistic regression with elastic net penalty
        (``SGDClassifier(loss="log", penalty="elasticnet")``).
    """
    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.intercept_ = 0.0
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        if (isinstance(self.precompute, six.string_types)
                and self.precompute == 'auto'):
            warnings.warn("Setting precompute to 'auto', was found to be "
                          "slower even when n_samples > n_features. Hence "
                          "it will be removed in 0.18.",
                          DeprecationWarning, stacklevel=2)
        # We expect X and y to be already float64 Fortran ordered arrays
        # when bypassing checks
        if check_input:
            y = np.asarray(y, dtype=np.float64)
            X, y = check_X_y(X, y, accept_sparse='csc', dtype=np.float64,
                             order='F',
                             copy=self.copy_X and self.fit_intercept,
                             multi_output=True, y_numeric=True)
            y = check_array(y, dtype=np.float64, order='F', copy=False,
                            ensure_2d=False)
        X, y, X_mean, y_mean, X_std, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=False)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or self.coef_ is None:
            coef_ = np.zeros((n_targets, n_features), dtype=np.float64,
                             order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_targets, dtype=np.float64)
        self.n_iter_ = []

        for k in xrange(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            _, this_coef, this_dual_gap, this_iter = \
                self.path(X, y[:, k],
                          l1_ratio=self.l1_ratio, eps=None,
                          n_alphas=None, alphas=[self.alpha],
                          precompute=precompute, Xy=this_Xy,
                          fit_intercept=False, normalize=False, copy_X=True,
                          verbose=False, tol=self.tol, positive=self.positive,
                          X_mean=X_mean, X_std=X_std, return_n_iter=True,
                          coef_init=coef_[k], max_iter=self.max_iter,
                          random_state=self.random_state,
                          selection=self.selection,
                          check_input=False)
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]

        self.coef_, self.dual_gap_ = map(np.squeeze, [coef_, dual_gaps_])
        self._set_intercept(X_mean, y_mean, X_std)

        # return self for chaining fit and predict calls
        return self

    @property
    def sparse_coef_(self):
        """ sparse representation of the fitted coef """
        return sp.csr_matrix(self.coef_)

    @deprecated(" and will be removed in 0.19")
    def decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : array, shape (n_samples,)
            The predicted decision function
        """
        return self._decision_function(X)

    def _decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : array, shape (n_samples,)
            The predicted decision function
        """
        check_is_fitted(self, 'n_iter_')
        if sp.isspmatrix(X):
            return np.ravel(safe_sparse_dot(self.coef_, X.T, dense_output=True) + self.intercept_)
        else:
            return super(ElasticNet, self)._decision_function(X)


class Lasso(ElasticNet):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

    Read more in the :ref:`User Guide <lasso>`.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` is with the Lasso object is not advised
        and you should prefer the LinearRegression object.

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.
        WARNING : The ``'auto'`` option is deprecated and will
        be removed in 0.18.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : int | array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.Lasso(alpha=0.1)
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    >>> print(clf.coef_)
    [ 0.85  0.  ]
    >>> print(clf.intercept_)
    0.15

    See also
    --------
    lars_path
    lasso_path
    LassoLars
    LassoCV
    LassoLarsCV
    sklearn.decomposition.sparse_encode

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    """
    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super(Lasso, self).__init__(
            alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)


def coefficient_matrix(dates, avg_days_yr, num_coefficients):
    """
    Fourier transform function to be used for the matrix of inputs for
    model fitting

    Args:
        dates: list of ordinal dates
        num_coefficients: how many coefficients to use to build the matrix

    Returns:
        Populated numpy array with coefficient values
    """
    w = 2 * np.pi / avg_days_yr

    matrix = np.zeros(shape=(len(dates), 7), order='F')

    matrix[:, 0] = dates
    matrix[:, 1] = np.cos(w * dates)
    matrix[:, 2] = np.sin(w * dates)

    if num_coefficients >= 6:
        matrix[:, 3] = np.cos(2 * w * dates)
        matrix[:, 4] = np.sin(2 * w * dates)

    if num_coefficients == 8:
        matrix[:, 5] = np.cos(3 * w * dates)
        matrix[:, 6] = np.sin(3 * w * dates)

    return matrix


def new_predictions():
    fmt              = "%m/%d/%Y"
    FileName         = 'AnimationSite_2500_657(Observation).csv'
    table            = pd.read_csv(FileName, sep=',', header=None)
    #print(table.shape)
    Time             = table.values[:, 0]
    dates            = np.asarray([datetime.strptime(t, fmt).toordinal() for t in Time])
    spectra_obs      = table.values[:, 7] # band 5
    max_iter         = 1000
    avg_days_yr      = 365.25
    num_coefficients = 8

    coef_matrix = coefficient_matrix(dates, avg_days_yr, num_coefficients)

    lasso = Lasso(max_iter=max_iter)
    model = lasso.fit(coef_matrix, spectra_obs)

    predictions = model.predict(coef_matrix)
    print(predictions)
    return True

