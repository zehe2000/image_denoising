import numpy as np
from gd_and_bfgs.fun import f, f_grad



def gradient_descent(img, x, smoothness = 0.5):
    """ Optimization with gradient descent and backtracking line search
        Plots intermediate results (image, stepwidth, energies)
    Args:
        img (np.ndarray): noisy image
        smoothness (float): weight of the smoothness term

    returns:
        energy: value of f() for the denoised image after 200 iterations """
        
    #initialize x
    x = img.astype(np.float64)
    y = img.astype(np.float64)
    
    tau = 1 # stepwidth
    beta = 0.9 #reduction factor for stepwidth
    delta = 0.5 # backtracking factor for armijo condition
    epsilon = 1e-8 # tolerance
    iter = 200 # number of iterations

    
    for i in range(iter):
        d_k = -f_grad(x, img, smoothness)
        #armijo condition for backtracking line search
        while not (f(x+tau*d_k, img, smoothness)[0] <= f(x, img, smoothness) + delta * tau * np.dot(f_grad(x, img, smoothness).T, d_k).all()) and tau > epsilon:
            tau *= beta
        #gradient descent update
        x = x + tau * d_k
        
    return x, f(img, x)[0]