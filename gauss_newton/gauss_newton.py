import numpy as np
from fun import f_res, grad_f


def optimize_gn(x, h, eps=1e-4):
    """ Finds the optimal function parameters using the Gauss-Newton method.
    Args:
        x (np.ndarray [10]): x-values
        h (np.ndarray [10]): observed h-values
        eps (float): tolerance value for termination

    Returns:
        x_opt (np.ndarray [2]): optimal parameter values
        iter_cnt (int): number of iterations required for convergence
    """
    
    p_k = np.asarray([1, 1]) #initial guess
    p_prev = p_k #previous guess
    tau = 1.0 #stepwidth
    iter_cnt = 0
    
    while True:
        #Hessian matrix 
        A = jacobian(x, p_k[0], p_k[1]).T.matmul(jacobian(x, p_k[0], p_k[1]))
        # negative gradient of f 
        b = -jacobian(x, p_k[0], p_k[1]).T.matmul(f_res(x, h, p_k[0], p_k[1]))
        #solve the linear system
        d = np.linalg.solve(A, b)
        #update the parameters
        p_k = p_k + tau * d
        if np.linalg.norm(p_k - p_prev) < eps:
            break
        p_prev = p_k
        iter_cnt += 1
        
    return p_k, iter_cnt

def jacobian(x, p1, p2):
    """ Computes the jacobian matrix
    Args:
        x (np.ndarray [10]): x-values
        p1, p2 (float): function parameters to optimize

    Returns:
        jac (np.ndarray [10, 2]): jacobian matrix
    """
    jac = [grad_f(pt, p1, p2) for pt in x]
    return jac