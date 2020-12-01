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


def shuffle(x1: ARR, x2: ARR) -> tp.Tuple[ARR, ARR]:
    """Apply the same shuffle to the rows of two arrays.
    
    :param x1: array to be shuffled
    :type x1: ``numpy.ndarray``
    :param x2: array to be shuffled in the same order
    :type x2: ``numpy.ndarray``
    :return: shuffled arrays
    :rtype: ``tuple`` of ``numpy.ndarray``
    """
    assert x1.shape[0] == x2.shape[0], f"Arrays 'x1' and 'x2' should have same number of rows."
    
    m = x1.shape[0]
    ordering = np.random.permutation(m)
    return x1[ordering], x2[ordering]


def train_test_split(X: ARR, y: ARR, ratio: float = 0.8) -> tp.Tuple[ARR, ARR, ARR, ARR]:
    """
    
    :param X: feature variables matrix
    :type X: ``numpy.ndarray``
    :param y: target variable vector
    :type y: ``numpy.ndarray``
    :param ratio: size of train data between 0 and 1 (default = 0.8)
    :type ratio: ``float``
    :return: x_train, y_train, x_test, y_test
    :rtype: ``tuple`` of ``numpy.ndarray``
    """
    assert X.shape[0] == y.shape[0], f"Arrays 'X' and 'y' should have same number of rows."
    
    split = round(X.shape[0] * ratio)
    return X[:split, :], y[:split, :], X[split:, :], y[split:, :]

