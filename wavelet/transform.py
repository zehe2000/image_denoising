import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from utils import resize_image_to_power_of_2, sign

def haar_1d(f: np.ndarray, endsize: int) -> None:
    """
    One-dimensional Haar wavelet transform (in place).
    """
    n = f.size
    h = 1.0 / np.sqrt(2.0)
    size = n
    while size >= 2 * endsize:
        u = f[:size].copy()
        new_size = size // 2
        for j in range(new_size):
            f[j] = h * (u[2 * j] + u[2 * j + 1])
            f[j + new_size] = h * (u[2 * j] - u[2 * j + 1])
        size = new_size

def haar_back_1d(f: np.ndarray, endsize: int) -> None:
    """
    One-dimensional Haar wavelet backtransformation (in place).
    """
    n = f.size
    h = 1.0 / np.sqrt(2.0)
    size = endsize
    while size <= n // 2:
        u = f[:2 * size].copy()
        for j in range(size):
            f[2 * j] = h * (u[j] + u[j + size])
            f[2 * j + 1] = h * (u[j] - u[j + size])
        size *= 2

def wavelet(f: np.ndarray, endsize: int) -> None:
    """
    2D tensor-product Haar wavelet transform (in place) on a square image.
    """
    n = f.shape[0]
    k = n
    while k >= 2 * endsize:
        # Haar transform in x direction (rows)
        for i in range(k):
            row = f[i, :k].copy()
            haar_1d(row, k // 2)
            f[i, :k] = row
        # Haar transform in y direction (columns)
        for j in range(k):
            col = f[:k, j].copy()
            haar_1d(col, k // 2)
            f[:k, j] = col
        k //= 2




def wavelet_back(f: np.ndarray, endsize: int) -> None:
    """
    Inverse 2D Haar wavelet transform (in place).
    """
    n = f.shape[0]
    k = 2 * endsize
    while k <= n:
        # Inverse Haar transform in x direction (rows)
        for i in range(k):
            row = f[i, :k].copy()
            haar_back_1d(row, k // 2)
            f[i, :k] = row
        # Inverse Haar transform in y direction (columns)
        for j in range(k):
            col = f[:k, j].copy()
            haar_back_1d(col, k // 2)
            f[:k, j] = col
        k *= 2

def shrink(f: np.ndarray, endsize: int, stype: int, T: float) -> int:
    """
    Perform wavelet shrinkage on the 2D wavelet coefficient array f.
    """
    n = f.shape[0]
    count = 0
    if stype == 0:  # Hard shrinkage
        for i in range(n):
            for j in range(n):
                if (i >= endsize) or (j >= endsize):
                    if abs(f[i, j]) < T:
                        f[i, j] = 0.0
                        count += 1
    elif stype == 1:  # Soft shrinkage
        for i in range(n):
            for j in range(n):
                if (i >= endsize) or (j >= endsize):
                    abs_val = abs(f[i, j])
                    if abs_val < T:
                        f[i, j] = 0.0
                        count += 1
                    else:
                        f[i, j] = sign(f[i, j]) * (abs_val - T)
    elif stype == 2:  # Nonnegative garrote shrinkage
        for i in range(n):
            for j in range(n):
                if (i >= endsize) or (j >= endsize):
                    abs_val = abs(f[i, j])
                    if abs_val < T:
                        f[i, j] = 0.0
                        count += 1
                    else:
                        f[i, j] = f[i, j] * (1.0 - (T * T) / (abs_val * abs_val))
    return count



