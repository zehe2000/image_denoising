import numpy as np
from fun import f, f_grad

def gradient_descent(img, smoothness = 0.5):
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
    
    tau = 10 # stepwidth before 1
    beta = 0.9 #reduction factor for stepwidth
    delta = 1e-4 # backtracking factor for armijo condition before 0.5
    epsilon = 1e-8 # tolerance
    iter = 200 # number of iterations

    
    for i in range(iter):
        d_k = -f_grad(x, y, smoothness)
        #armijo condition for backtracking line search
        # Precompute the gradient and its dot product with d_k
        grad_x = f_grad(x, img, smoothness)
        dot_val = np.sum(grad_x * d_k)  # scalar

        while (
            f(x + tau * d_k, y, smoothness)[0]
            > f(x, y, smoothness)[0] + delta * tau * dot_val
        ) and (tau > epsilon):
            tau *= beta


        #gradient descent update
        x = x + tau * d_k
        
    return x, f(x, img, smoothness)[0]