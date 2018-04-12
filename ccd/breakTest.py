
# Functions used to test for breaks

# Order of dimensions are expected to always be: bands, time, coefficients

import numpy as np

import os.path


# These values still need to be updated at some point, but they should be roughly correct
# Also need documentation if this version of PyCCD is used
# Also would be good to add p-value labels to the file
#def readCutoffsFromFile():
#    cutoffs = np.loadtxt(os.path.join(os.path.dirname(__file__), 'StatsCurrent.txt'))
#    return cutoffs



def breakTestIncludingModelError(compareObservationResiduals,regressorsForCompareObservations,
        msrOfCurrentModels,nObservationsInModel,cutoffLookupTable,pValueForBreakTest,inverseMatrixXTX):
    """Test for model breaks; include observation error and model error.
    Also allows for a variable number of comparison observations.
    """

    nBands,nCompareObservations = compareObservationResiduals.shape
    nCoefficients = regressorsForCompareObservations.shape[1]
    magnitudes = np.zeros((nCompareObservations))

#    inverseMatrixXTX = np.linalg.inv(matrixXTX)

    for i in range(nCompareObservations):

        xAtThisTime = np.copy(regressorsForCompareObservations[i,:])
        # This is the additional factor to account for model error
        modelErrorAdjustment = np.matmul(np.matmul(np.transpose(xAtThisTime),inverseMatrixXTX),xAtThisTime)

        for band in range(nBands):
            magnitudes[i] += np.power(compareObservationResiduals[band,i],2)/(msrOfCurrentModels[band]*(1+modelErrorAdjustment))

    # The input p-value is for the entire test, including all nCompareObservations points. The p-value for the minimum doesn't
    #    need to be as stringent, since we know that all the other observations have larger residuals. The chance that all of
    #    the compare observations have p<individualPValue is individualPValue^nCompareObservations
    individualPValue = np.power(pValueForBreakTest,1/nCompareObservations)

    nDegreesOfFreedom = nObservationsInModel-nCoefficients
    cutoff = cutoffLookupTable[501-int(individualPValue*1000),min(nDegreesOfFreedom,cutoffLookupTable.shape[1])-1]

    if min(magnitudes) > cutoff:
        return True, magnitudes
    else:
        return False, magnitudes

