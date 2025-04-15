import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from transform import *
from utils import resize_image_to_power_of_2, sign


if __name__ == "__main__":
    # 1. Load image and convert to grayscale
    image = Image.open("../puppy.png")
    
    # 2. Resize to a power-of-two width (and possibly crop to square if needed)
    image_resized = resize_image_to_power_of_2(image)
    # Crop to a square if you want a full 2D transform on an n x n image:
    min_dim = min(image_resized.size)
    image_square = image_resized.crop((0, 0, min_dim, min_dim))
    
    # 3. Convert to float to avoid overflow
    image_matrix = np.array(image_square).astype(np.float64)
    
    # 4. Forward 2D wavelet transform in place
    #    endsize=1 usually decomposes fully for a power-of-2 image
    wavelet(image_matrix, endsize=1)
    
    # 5. Shrink (i.e., threshold) the wavelet coefficients to remove noise
    #    stype = 1 => soft thresholding, pick T based on noise
    threshold = 1.0  # Try different values; or compute from noise estimate
    shrinkage_type = 1
    shrink(image_matrix, endsize=1, stype=shrinkage_type, T=threshold)
    
    # 6. Inverse 2D wavelet transform to reconstruct denoised image
    wavelet_back(image_matrix, endsize=1)
    denoised_matrix = image_matrix  # The final denoised image (float64)
    
    # 7. Display original vs. denoised
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1,2,1)
    plt.imshow(image_square, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.imshow(denoised_matrix, cmap='gray')
    plt.title("Denoised Image (Wavelet Shrinkage)")
    plt.axis("off")
    
    plt.show()