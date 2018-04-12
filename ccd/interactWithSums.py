
# Functions to interact with sum matrices used in fitting of CCD models

# Order of dimensions are expected to always be: bands, time, coefficients


# matrixXTX - Gram matrix; transpose(X)*X; is symmetrical; [nCoefficients, nCoefficients]
# vectorsXTY - contains transpose(X)*Y for all bands in Y; [nBands, nCoefficients]


# Inputs:
#    X - Design matrix, matrix of regressors; [nObservations, nCoefficients]
#    Y - Satellite reflectance values for a pixel; [nBands, nObservations]


# Matrices are calculated (all sums are across time) as:
#    matrixXTX[i,j] = sum(X[timeRange,i]*X[timeRange,j])
#    vectorsXTY[band,i] = sum(X[timeRange,i]*y[band,timeRange])
#    sumX[i] = sum(X[timeRange,i])
#    sumY[band] = sum(y[band,timeRange])
#    sumYSquared[band] = sum(y[band,timeRange]^2)


import numpy as np


#def incrementSums(indexToAdd,X,Y,matrixXTX,vectorsXTY,sumX,sumY,sumYSquared):
def incrementSums(indexToAdd,X,Y,matrixXTX,vectorsXTY,sumYSquared):
    """Increment the sum matrices (matrixXTX,vectorsXTY,sumX,sumY,sumYSquared) using X and y as input
    indexToAdd is the index (out of nObservations) at which to take values from X and y
    """
    nBands,nCoefficients = vectorsXTY.shape

    # XTX
    for i in range(nCoefficients):
        for j in range(nCoefficients):
            matrixXTX[i,j] += X[indexToAdd,i]*X[indexToAdd,j]

    # XTY
    for band in range(nBands):
        for i in range(nCoefficients):
            vectorsXTY[band,i] += X[indexToAdd,i]*Y[band,indexToAdd]

    # sum X
#    for i in range(nCoefficients):
#        sumX[i] += X[indexToAdd,i]

    # sum Y and sum Y^2
    for band in range(nBands):
#        sumY[band] += Y[band,indexToAdd]
        sumYSquared[band] += Y[band,indexToAdd]**2

def incrementXTX(indexToAdd,X,Y,matrixXTX):
    nCoefficients = matrixXTX.shape[0]
    for i in range(nCoefficients):
        for j in range(nCoefficients):
            matrixXTX[i,j] += X[indexToAdd,i]*X[indexToAdd,j]

def createSumArrays(nBands, nCoefficients):
    matrixXTX = np.zeros((nCoefficients,nCoefficients),order='C')
    vectorsXTY = np.zeros((nBands,nCoefficients),order='F')
#    sumX = np.zeros((nCoefficients),order='F')
#    sumY = np.zeros((nBands),order='F')
    sumYSquared = np.zeros((nBands),order='F')
#    return matrixXTX, vectorsXTY, sumX, sumY, sumYSquared
    return matrixXTX, vectorsXTY, sumYSquared

def createXTX(nCoefficients):
    matrixXTX = np.zeros((nCoefficients,nCoefficients),order='C')
    return matrixXTX

def centerSumMatrices(matrixXTX, vectorsXTY, sumX, sumY, sumYSquared, nObservationsInMatrices):
    """ Center the sum matrices so that they have 0 intercept
        Acts on matrixXTX, vectorsXTY, and sumYSquared
        Algebra for element i,j in XTX (LaTeX):
            \sum_t^n{(x_{t,i}-\bar{x_i})(x_{t,j}-\bar{x_j})} =
            \sum_t^n{x_{t,i}x_{t,j}}-\bar{x_i}\sum_t^n x_{t,j}-\bar{x_j}\sum_t^n x_{t,i}+n \bar{x_i} \bar{x_j} =
            \sum_t^n{x_{t,i}x_{t,j}}-\frac{\sum_t^n x_{t,i}\sum_t^n x_{t,j}}{n}
        Similar for XTY and Y^2
    """
    nBands,nCoefficients = vectorsXTY.shape

    # XTX
    for i in range(nCoefficients):
        for j in range(nCoefficients):
            matrixXTX[i,j] = (matrixXTX[i,j]-sumX[i]*sumX[j]/nObservationsInMatrices)

    # XTY and Y^2
    for band in range(nBands):
        for i in range(nCoefficients):
            vectorsXTY[band,i] = (vectorsXTY[band,i]-sumX[i]*sumY[band]/nObservationsInMatrices)
        sumYSquared[band] = sumYSquared[band]-np.power(sumY[band],2)/nObservationsInMatrices





def ssrForModelUsingMatrixXTX(modelBetas,matrixXTX,vectorsXTY,sumYSquared):
    """Computes sum of squared residuals for the fitted model over the sum range
    Algebraic manipulation (LaTeX):
        \sum_t{(y_t-\hat{y_t})^2} = \sum_t\left(y_t-\sum_i\beta_ix_{t,i}\right)^2 =
        \sum_t{y_t^2}-2\sum_i\left(\beta_i\sum_t y_t x_{t,i}\right)+
        \sum_i\left(\beta_i\sum_j\left(\beta_j\sum_t x_{t,i} x_{t,j}\right) \right)
    """
    # For running data from a single band
    if vectorsXTY.ndim==1:
        nCoefficients = vectorsXTY.shape[0]
        result = np.copy(sumYSquared)
        for i in range(nCoefficients):
            result -= 2*modelBetas[i]*vectorsXTY[i]
            sum = 0
            for j in range(nCoefficients):
                sum += modelBetas[j]*matrixXTX[i,j]
            result += modelBetas[i]*sum
    # Multiple bands at once
    else:
        nBands,nCoefficients = vectorsXTY.shape
        result = np.copy(sumYSquared)
        for band in range(nBands):
            for i in range(nCoefficients):
                result[band] -= 2*modelBetas[band,i]*vectorsXTY[band,i]
                sum = 0
                for j in range(nCoefficients):
                    sum += modelBetas[band,j]*matrixXTX[i,j]
                result[band] += modelBetas[band,i]*sum

    return result



