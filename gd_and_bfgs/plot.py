import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fun import f, f_grad  
from gradient_descent import gradient_descent
from bfgs import bfgs

# Load the noisy image 
noisy_image = Image.open("../puppy.png")
noisy_image = np.array(noisy_image)

# normalize or scale your image 
noisy_image = noisy_image / 255.0

# Apply gradient descent denoising.
denoised_image_gd, final_energy = gradient_descent(noisy_image, 0.5)

# Apply BFGS denoising.
denoised_image_bfgs, final_energy_bfgs = bfgs(noisy_image, 0.5)

# Plot the original and denoised images side by side.
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Plot the noisy image
axes[0].imshow(noisy_image, cmap='gray')
axes[0].set_title("Noisy Image")
axes[0].axis("off")

# Plot the denoised image
axes[1].imshow(denoised_image_gd, cmap='gray')
axes[1].set_title("Denoised Image using gradient descent")
axes[1].axis("off")

# Plot the BFGS denoised image
axes[2].imshow(denoised_image_bfgs, cmap='gray')
axes[2].set_title("Denoised Image using BFGS")
axes[2].axis("off")

plt.tight_layout()
plt.show()

