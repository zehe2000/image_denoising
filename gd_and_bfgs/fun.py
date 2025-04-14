

import numpy as np
from matplotlib import pyplot as plt

def f(x, y, alpha, bfgs=False):
    """
    Computes the energy for the optimization problem
    Args:
        x (np.ndarray [H, W] or [H*W,]): image that is optimized
        y (np.ndarray [H, W] or [H*W,]): noisy image
        alpha (float): weight of the smoothness term
        bfgs (bool): if True only return energy

    Returns:
        en_total (float): total energy value
        en_data (float): energy value of the data term
        en_smooth (float): energy value of the smoothness term
    """
    if x.ndim == 1: # reshape x to img dims
        x = x.reshape(60, 80)
    
    # Data term
    data_diff = x - y
    en_data = np.sum(np.sqrt(data_diff**2 + 1))
    
    # Smoothness term
    dx = x_dx(x)
    dy = x_dy(x)
    en_smooth = alpha * np.sum(np.sqrt(dx**2 + dy**2 + 1))
    
    # Total energy
    en_total = en_data + en_smooth
    
    if bfgs:
        return en_total
    else:
        return en_total, en_data, en_smooth
    
def x_dx(img: np.ndarray):
    """computes the derivative of the image in the x direction
    Args: 
        img (np.ndarray): image
    Returns:
        x_r (np.ndarray): derivative of the image in the x direction
    """
    assert img.ndim == 2
    x_r = np.zeros(img.shape)
    x_r[:,:-1] = img[:, 1:] # x_00 is never subtracted so we save this space
    x_r[:,-1] = img[:,-1] # last entry is mirrored
    x_dx = img - x_r
    return x_dx

def x_dy(img: np.ndarray):
    """computes the derivative of the image in the y direction
    Args:
        img (np.ndarray): image
    Returns:
        x_d (np.ndarray): derivative of the image in the y direction
    """
    assert img.ndim == 2
    x_b = np.zeros(img.shape)
    x_b[:-1,:] = img[1:, :] # same as x_dx but for j+1
    x_b[-1,:] = img[-1,:]
    x_dy = img - x_b
    return x_dy
    
        
    
def f_grad(x, y, alpha):
    """ Computes the gradient for the optimization problem
    Args:
        x (np.ndarray [60, 80] or [4800,]): image that is optimized
        y (np.ndarray [60, 80] or [4800,]): noisy image
        alpha (float): weight of the smoothness term

    Returns:
        gradient (np.ndarray [60, 80] or [4800,]): gradient matrix
    """
    
    # gradient in x direction
    get_x_dx = x_dx(x)
    
    #gradient in y direction
    get_x_dy = x_dy(x)
    
    grad = np.zeros_like(x)
    
    #derivative of the data term
    grad = (x - y) / np.sqrt((x - y)**2 + 1)
    
    undo = False
    if x.ndim == 1:
        x = x.reshape(60, 80)
        undo = True

    
    #derivative of the smoothness term
    denominator = np.sqrt(get_x_dx**2 + get_x_dy**2 + 1)
    grad += alpha * (get_x_dx + get_x_dy) / denominator
    
    grad1 = - alpha * get_x_dx / denominator
    grad2 = - alpha * get_x_dy / denominator
    
    grad[:, 1:] += grad1[:, 0 :-1]
    grad[1:, :] += grad2[0 :-1, :]
    
    if undo: # for scipy bfgs go back to 1 dim
        grad_smooth = grad_smooth.reshape(-1)
    
    return grad