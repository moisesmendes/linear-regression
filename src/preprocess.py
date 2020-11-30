__author__ = 'Moises Mendes'
__version__ = '0.1.0'
__all__ = [
    'feature_normalize',
    'add_intercept_term',
]

import typing as tp
import numpy as np

ARR = np.ndarray


def feature_normalize(X: ARR) -> tp.Tuple[ARR, np.float64, np.float64]:
    """Returns a normalized version of X with zero mean and standard deviation equals one.
    
    :param X: data with feature columns
    :type X: ``numpy.ndarray``
    :return: normalized data, mean and standard deviation of original data
    :rtype: ``tuple`` of ``numpy.ndarray``, ``numpy.float64``, ``numpy.float64``
    """
    mean = X.mean()
    std = X.std()
    X_norm = (X - mean)/std
    return X_norm, mean, std


def add_intercept_term(X: ARR) -> ARR:
    """Returns X feature matrix with intercept term (column of ones).
    
    :param X: Feature matrix.
    :type X: ``numpy.ndarray``
    :return: Feature matrix with intercept term.
    :rtype: ``numpy.ndarray``
    """
    return np.hstack((np.ones((X.shape[0], 1)), X))
