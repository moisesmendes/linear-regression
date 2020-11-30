__author__ = 'Moises Mendes'
__version__ = '0.1.0'
__all__ = [
    'quadratic_error',
]

import typing as tp

import numpy as np

ARR = np.ndarray


def quadratic_error(X: ARR, y: ARR, theta: ARR) -> tp.Tuple[ARR, ARR]:
    assert X.ndim == 2, f"'X' should have 2 dimensions, but has {X.ndim}"
    assert y.ndim == 2, f"'y' should have 2 dimensions, but has {y.ndim}"
    assert theta.ndim == 2, f"'theta' should have 2 dimensions, but has {theta.ndim}"

    m = X.shape[0]
    assert y.shape == (m, 1), f"shape of 'y' is {y.shape}; expected ({m}, 1)"
    
    n = X.shape[1]
    assert theta.shape == (n, 1), f"shape of 'theta' is {theta.shape}; expected ({n}, 1)"
    
    pred_error = X.dot(theta) - y
    cost = (np.dot(pred_error.T, pred_error)) / (2 * m)
    gradient = (np.dot(X.T, pred_error)) / m
    return cost, gradient
