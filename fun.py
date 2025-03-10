import numpy as np
from matplotlib import pyplot as plt

#This is our function:
#-Latex Code: f(x) = \sum_{i,j}\left ( \sqrt{(x_{i,j} - y_{i,j})^2 +1} + 
# \frac{1}{2} \sqrt{(x_{i,j} - x_{i+1,j})^2 + (x_{i,j} - x_{i,j+1})^2 +1} \right)

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
    H, W = y.shape
    x = x.reshape(H, W)
    
    # Data term
    data_diff = x - y
    en_data = np.sum(np.sqrt(data_diff**2 + 1))
    
    # Smoothness term
    dx = x_dx(x)
    dy = x_dy(x)
    en_smooth = np.sum(np.sqrt(dx**2 + dy**2 + 1))
    
    # Total energy
    en_total = en_data + (alpha / 2) * en_smooth
    
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
    x_r = np.zeros_like(img)
    x_r[:, :-1] = img[:, 1:]
    x_r[:, -1] = img[:, -1]
    return img - x_r

def x_dy(img: np.ndarray):
    """computes the derivative of the image in the y direction
    Args:
        img (np.ndarray): image
    Returns:
        x_d (np.ndarray): derivative of the image in the y direction
    """
    x_d = np.zeros_like(img)
    x_d[:-1, :] = img[1:, :]
    x_d[-1, :] = img[-1, :]
    return img - x_d
        
    
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
    x_dx = x_dx(x)
    
    #gradient in y direction
    x_dy = x_dy(x)
    
    grad = np.zeros_like(x)
    
    #derivative of the data term
    grad = (x - y) / np.sqrt((x - y)**2 + 1)
    
    #derivative of the smoothness term
    denominator = np.sqrt(x_dx**2 + x_dy**2 + 1)
    grad+= alpha * (x_dx + x_dy) / denominator
    
    grad1 = alpha * (x + x_dx) / denominator
    grad2= alpha * (x + x_dy) / denominator
    
    grad[:, 1:] -= grad1[:, :-1]
    grad[1:, :] -= grad2[:-1, :]
    
    return grad