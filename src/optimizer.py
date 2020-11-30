__author__ = 'Moises Mendes'
__version__ = '0.1.0'
__all__ = [
    'GradientDescent',
]

import typing as tp

import numpy as np

ARR = np.ndarray
TARR = tp.Tuple[ARR, ARR]
ARR_FUNC = tp.Callable[[ARR, ARR, ARR], TARR]


class GradientDescent:
    """Optimizer class implementing the Gradient Descent algorithm.
    It minimizes the cost function taking steps in the opposite direction of
    the gradient of the cost function.
    
    Methods:
        - optimize: run gradient descent for given inputs and cost function.
    """
    
    def __init__(self, alpha, max_iter, tolerance):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def optimize(self, X: ARR, y: ARR, theta: ARR, cost_function: ARR_FUNC) -> TARR:
        """Runs gradient descent method until one of stop conditions is reachead:
        1. maximum number of iterations;
        2. change in cost function smaller than tolerance.
        
        :param X: Feature variables matrix, where each row is one training example.
        :type X: ``numpy.ndarray``
        :param y: Target variable vector.
        :type y: ``numpy.ndarray``
        :param theta: Model parameters.
        :type theta: ``numpy.ndarray``
        :param cost_function: Implements cost function and its gradient given X, y and theta.
        :type cost_function: ``callable``
        :return: Final model parameters and history of cost function at every iteration.
        :rtype: ``tuple`` of ``numpy.ndarray``, ``numpy.ndarray``
        """
        cost_history = np.zeros(self.max_iter, dtype=np.float64)
        
        for i in range(0, self.max_iter):
            cost, gradient = cost_function(X, y, theta)
            theta = theta - self.alpha * gradient
            cost_history[i] = cost
            
            if (i > 1) and (abs(cost_history[i-1] - cost_history[i]) < self.tolerance):
                print(f'stop by tolerance criteria: {cost_history[i] - cost_history[i-1]} < {self.tolerance}')
                cost_history = cost_history[:i+1]
                break

        print(f'End of optimize: iter {i+1} - cost {cost} - theta {theta.T}')
        return theta, cost_history
