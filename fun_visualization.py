import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
from fun import f


def f_visualize(image_input, alpha=1.0, pixel_indices=((50,50), (50,51)), grid_range=20):
    """
    Visualizes a slice of the energy function using an input image as the noisy image.
    
    Args:
        image_input (str or np.ndarray): path to the image file or an image array.
        alpha (float): weight for the smoothness term.
        pixel_indices (tuple): a tuple containing two tuples for the pixel locations to vary.
        grid_range (float): range around the original pixel values to vary.
        
    Steps:
      1. Load the image if a file path is provided; otherwise use the provided array.
      2. Convert the image to grayscale if needed.
      3. Use the image as the noisy image y and initialize x = y.
      4. Vary the two specified pixel values over a grid.
      5. Compute the total energy for each combination.
      6. Plot the resulting 3D surface.
    """
    # Load the image if image_input is a file path.
    if isinstance(image_input, str):
        img = imread(image_input)
    else:
        img = image_input

    # Convert to grayscale if the image is RGB
    if img.ndim == 3:
        y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        y = img.copy()
    
    # Ensure the image is of type float for computation
    y = y.astype(np.float64)
    
    # Use y as the noisy image and initialize x to y
    x_fixed = y.copy()
    
    # Unpack the pixel indices to vary.
    (i1, j1), (i2, j2) = pixel_indices
    
    # Define grid around the original pixel values for the two chosen pixels.
    val1_center = x_fixed[i1, j1]
    val2_center = x_fixed[i2, j2]
    
    grid_vals1 = np.linspace(val1_center - grid_range, val1_center + grid_range, 100)
    grid_vals2 = np.linspace(val2_center - grid_range, val2_center + grid_range, 100)
    X, Y_grid = np.meshgrid(grid_vals1, grid_vals2)
    
    Z = np.zeros_like(X)
    
    # Evaluate the energy for each candidate pair
    for a in range(X.shape[0]):
        for b in range(X.shape[1]):
            x_candidate = x_fixed.copy()
            x_candidate[i1, j1] = X[a, b]
            x_candidate[i2, j2] = Y_grid[a, b]
            en_total, _, _ = f(x_candidate, y, alpha)
            Z[a, b] = en_total
    
    # Plot the energy surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel(f'Pixel ({i1},{j1}) Value')
    ax.set_ylabel(f'Pixel ({i2},{j2}) Value')
    ax.set_zlabel('Total Energy')
    ax.set_title('Energy Function Slice using Input Image Puppy')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    
if __name__ == '__main__':
    # Load the image
    image_path = 'puppy.png'
    
    f_visualize(image_path)