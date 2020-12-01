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

    def optimize_with_validation(self,
                                 x_train: ARR, y_train: ARR,
                                 x_val: ARR, y_val: ARR,
                                 theta: ARR, cost_function: ARR_FUNC) -> tp.Tuple[ARR, ARR, ARR]:
        """Runs gradient descent method until one of stop conditions is reachead:
        1. maximum number of iterations;
        2. change in cost function smaller than tolerance.

        :param x_train: Training feature variables matrix.
        :type x_train: ``numpy.ndarray``
        :param y_train: Training target variable vector.
        :type y_train: ``numpy.ndarray``
        :param x_val: Validation feature variables matrix.
        :type x_val: ``numpy.ndarray``
        :param y_val: Validation target variable vector.
        :type y_val: ``numpy.ndarray``
        :param theta: Model parameters.
        :type theta: ``numpy.ndarray``
        :param cost_function: Implements cost function and its gradient given x_train, y_train and theta.
        :type cost_function: ``callable``
        :return: Final model parameters and history of cost function for training and validation.
        :rtype: ``tuple`` of ``numpy.ndarray``, ``numpy.ndarray``
        """
        train_cost = np.zeros(self.max_iter, dtype=np.float64)
        val_cost = np.zeros(self.max_iter, dtype=np.float64)
    
        for i in range(0, self.max_iter):
            train_cost[i], gradient = cost_function(x_train, y_train, theta)
            val_cost[i], _ = cost_function(x_val, y_val, theta)
            
            theta = theta - self.alpha * gradient
            
            if (i > 1) and (abs(train_cost[i - 1] - train_cost[i]) < self.tolerance):
                print(f'stop by tolerance criteria: {train_cost[i] - train_cost[i - 1]} < {self.tolerance}')
                train_cost = train_cost[:i + 1]
                val_cost = val_cost[:i + 1]
                break
    
        print(f'End of optimize: iter {i+1} - train cost {train_cost[i]} / val cost: {val_cost[i]}')
        print(f'- theta {theta.T}')
        return theta, train_cost, val_cost
