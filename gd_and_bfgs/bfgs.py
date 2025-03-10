import numpy as np
from gd_and_bfgs.fun import f, f_grad
from scipy.optimize import minimize


def bfgs(img, smoothness):
    """ BFGS method from scipy optimize
        Plots the denoised image and prints intermediate results

    Args:
        img (np.ndarray): noisy image
        smoothness(float): weight of the smoothness term, 0.5 in this exercise

    returns:
        energy: value of f() for the denoised image after tolerance = 0.02
                is reached
    """
    #initialize x
    x = img.astype(np.float64)
    y = img.astype(np.float64)
    
    #energy function
    fun = lambda x: f(x, y, smoothness, bfgs=True) 
    #gradient of the energy function
    jac = lambda x: f_grad(x, y.flatten(), smoothness) 
    
    #minimize the energy function
    res = minimize(fun, x, jac=jac, method='BFGS', tol=0.02,
                   options={'disp': True})
    
    # the shape of y is 60x80, so we reshape the result to this shape
    res_img = res.x.reshape(y.shape)
    
    return res_img, f(res_img, y, smoothness)[0]