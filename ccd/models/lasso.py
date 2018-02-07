from sklearn import linear_model
from sklearn.linear_model.coordinate_descent import ElasticNet,enet_path
import numpy as np
from cachetools import LRUCache

from ccd.models import FittedModel
from ccd.math_utils import calc_rmse

from ccd.interactWithSums import ssrForModelUsingMatrixXTX

cache = LRUCache(maxsize=1000)


def __coefficient_cache_key(observation_dates):
    return tuple(observation_dates)


# @cached(cache=cache, key=__coefficient_cache_key)
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

    # lookup optimizations
    # Before optimization - 12.53% of total runtime
    # After optimization  - 10.57% of total runtime
    cos = np.cos
    sin = np.sin

    w12 = w * dates
    matrix[:, 0] = dates
    matrix[:, 1] = cos(w12)
    matrix[:, 2] = sin(w12)

    if num_coefficients >= 6:
        w34 = 2 * w12
        matrix[:, 3] = cos(w34)
        matrix[:, 4] = sin(w34)

    if num_coefficients >= 8:
        w56 = 3 * w12
        matrix[:, 5] = cos(w56)
        matrix[:, 6] = sin(w56)

    return matrix


def fitted_model(dates, spectra_obs, max_iter, avg_days_yr, num_coefficients):
    """Create a fully fitted lasso model.

    Args:
        dates: list or ordinal observation dates
        spectra_obs: list of values corresponding to the observation dates for
            a single spectral band
        num_coefficients: how many coefficients to use for the fit
        max_iter: maximum number of iterations that the coefficients
            undergo to find the convergence point.

    Returns:
        sklearn.linear_model.Lasso().fit(observation_dates, observations)

    Example:
        fitted_model(dates, obs).predict(...)
    """
    coef_matrix = coefficient_matrix(dates, avg_days_yr, num_coefficients)

    lasso = linear_model.Lasso(max_iter=max_iter)
    model = lasso.fit(coef_matrix, spectra_obs)

    predictions = model.predict(coef_matrix)
    rmse, residuals = calc_rmse(spectra_obs, predictions)

    return FittedModel(fitted_model=model, rmse=rmse, residual=residuals)


def predict(model, dates, avg_days_yr):
    coef_matrix = coefficient_matrix(dates, avg_days_yr, 8)

    return model.fitted_model.predict(coef_matrix)



def fitted_model_using_sums(X, y, max_iter, matrixXTXcentered, vectorsXTYcentered, sumYSquaredCentered, meanX, meanY, normX, calculateResiduals):
    """ Fit the Lasso model, using the precomputed matrices
    Sum matrices are assumed to be already centered
    If calculateResiduals is set, residuals are calculated and the RMSE is calculated from the residuals; otherwise the
    RMSE is calculated based on the precomputed matrices.
    """

    lasso = LassoFromMatricesForCCD(precompute=matrixXTXcentered,max_iter=max_iter)
    model = lasso.fit(X, y, vectorsXTYcentered, meanX, meanY, normX)

    # Calculate out the residuals and the RMSE. This RMSE is calculated based on N, not using degrees of freedom.
    if calculateResiduals is True:
        predictions = model.predict(X)
        rmse, residuals = calc_rmse(y, predictions)
    # Calculate the RMSE using the matrices. This RMSE is based on degrees of freedom.
    else:
        modelBetas = np.array([float(c) for c in model.coef_])
        modelSSR = ssrForModelUsingMatrixXTX(modelBetas,matrixXTXcentered,vectorsXTYcentered,sumYSquaredCentered)
        rmse = np.sqrt(modelSSR/(X.shape[0]-X.shape[1]-1))
        residuals = np.nan

    return FittedModel(fitted_model=model, rmse=rmse, residual=residuals)




class LassoFromMatricesForCCD(ElasticNet):
    """ See class Lasso in scikit-learn for more information
    """
    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0,
                 copy_X=True, max_iter=1000,precompute=False,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super(LassoFromMatricesForCCD, self).__init__(
            alpha=alpha, l1_ratio=1.0, fit_intercept=True,
            normalize=False, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state, selection=selection)

    def fit(self, X, y, vectorsXTYcentered, X_offset, y_offset, X_scale):
        """ Modified from ElasticNet class in scikit-learn
        Fit the lasso model using precalculated matrices
        The inputs with nTime dimensions (X and y) are theoretically only used for getting dimensions/type and dotting the y
        Would need to modify "enet_path" to pass in sumYSquaredCentered instead of y
        X - the design matrix [nTime, nCoefficients]
        y - the satellite reflectance values [nBands, nTime]
        vectorsXTYcentered - the XTY vectors grouped together in a numpy array (assumed to be centered) [nBands, nCoefficients]
        X_offset - vector with the mean of X[:,i], also is the offset used to center XTX and XTY [nCoefficients]
        y_offset - the mean of y[i,:], offset used to center XTY [nBands]
        X_scale - the factor used to scale the X values in XTX and XTY [nCoefficients]
        output coef_ size is [nBands, nCoefficients]
        """

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

#        if isinstance(self.precompute, six.string_types):
#            raise ValueError('precompute should be one of True, False or'
#                             ' array-like. Got %r' % self.precompute)

        # We expect X and y to be float64 Fortran ordered arrays
        # when bypassing checks
#        if check_input:
#            X, y = check_X_y(X, y, accept_sparse='csc',
#                             order='F', dtype=[np.float64],
#                             copy=self.copy_X and self.fit_intercept,
#                             multi_output=True, y_numeric=True)
#            y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
#                            ensure_2d=False)

#        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
#                _pre_fit(X, y, None, self.precompute, self.normalize,
#                self.fit_intercept, copy=False)

        # Switch around the order of the dimensions from the original ElasticNet class code
        if y.ndim == 1:
            y = y[np.newaxis, :]
        if vectorsXTYcentered is not None and vectorsXTYcentered.ndim == 1:
            vectorsXTYcentered = vectorsXTYcentered[np.newaxis, :]

        n_bands, n_features = vectorsXTYcentered.shape

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros((n_bands, n_features), dtype=X.dtype,
                             order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_bands, dtype=X.dtype)
        self.n_iter_ = []
        for k in range(n_bands):
            this_Xy = np.ascontiguousarray(vectorsXTYcentered[k, :])
            _, this_coef, this_dual_gap, this_iter = \
                self.path(X, y[k, :],
                          l1_ratio=self.l1_ratio, eps=None,
                          n_alphas=None, alphas=[self.alpha],
                          precompute=self.precompute, Xy=this_Xy,
                          fit_intercept=False, normalize=False, copy_X=True,
                          verbose=False, tol=self.tol, positive=self.positive,
                          X_offset=X_offset, X_scale=X_scale, return_n_iter=True,
                          coef_init=coef_[k], max_iter=self.max_iter,
                          random_state=self.random_state,
                          selection=self.selection, check_input=False)
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        if n_bands == 1:
            self.n_iter_ = self.n_iter_[0]

        self.coef_, self.dual_gap_ = map(np.squeeze, [coef_, dual_gaps_])
        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self

