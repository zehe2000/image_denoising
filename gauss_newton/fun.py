import numpy as np
from matplotlib import pyplot as plt

#our function: y = p1*e^(p2*x)

def f(x, p1, p2):
    """ Computes the values of the parametric curve we want to fit to the data points.
    Args:
        x (np.ndarray [10]): x-values
        p1, p2 (float): function parameters to optimize

    Returns:
        y (np.ndarray [10]): predicted y-values
    """
    y = p1 * np.exp(p2 * x)
    return y

def f_res(x, h, p1, p2):
    """ Computes the residual between the predicted y-values and the observed h-values.
    Args:
        x (np.ndarray [10]): x-values
        h (np.ndarray [10]): observed h-values
        p1, p2 (float): function parameters to optimize

    Returns:
        r (np.ndarray [10]): residuals
    """
    r = f(x, p1, p2) - h
    return r

def grad_f(x, p1, p2):
    """ Computes the gradient of the residual w.r.t. p1 and p2.
    Args:
        x (np.ndarray [10]): x-values
        p1, p2 (float): function parameters to optimize

    Returns:
        grad (np.ndarray [2]): two-dimensional gradient vector
    """
    grad = np.asarray((np.exp(p2 * x), p1 * x * np.exp(p2 * x)))
    return grad.astype(np.float32)
