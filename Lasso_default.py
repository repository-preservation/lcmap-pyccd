from sklearn import linear_model
import numpy as np
from cachetools import LRUCache
import pandas as pd
from datetime import datetime
# import statsmodels.api as sm

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


def predictions():
    fmt="%m/%d/%Y"
    FileName = 'AnimationSite_2500_657(Observation).csv'
    table = pd.read_csv(FileName, sep=',',header=None)
    print(table.shape)
    Time=table.values[:,0]
    dates = np.asarray([datetime.strptime(t,fmt).toordinal() for t in Time])
    spectra_obs=table.values[:,7] # band 5
    max_iter=1000
    avg_days_yr=365.25
    num_coefficients=8

    coef_matrix = coefficient_matrix(dates, avg_days_yr, num_coefficients)

    lasso = linear_model.Lasso(max_iter=max_iter)
    model = lasso.fit(coef_matrix, spectra_obs)

    predictions = model.predict(coef_matrix)
    print(predictions)
    return True

