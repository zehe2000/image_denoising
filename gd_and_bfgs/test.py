#from gradient_descent import gradient_descent
#from fun import f, f_grad, x_dx, x_dy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("../puppy.png")

def get_x_dx(img: np.ndarray):
  """"
    Calculates the term (x_i,j - x_i+1,j) for all values of the input
    Args:
        img (np.ndarray [60, 80]): image
    Returns:
        x_dx (np.ndarray [60, 80]): difference with the right neighbor
  """
  assert img.ndim == 2
  x_r = np.zeros(img.shape)
  x_r[:,:-1] = img[:, 1:] # x_00 is never subtracted so we save this space
  x_r[:,-1] = img[:,-1] # last entry is mirrored
  x_dx = img - x_r
  # Alternatively
  # kernel = np.asarray([[-1,1]])
  # x_dx = scipy.ndimage.convolve(x, kernel, mode='nearest')
  return x_dx

#test get_x_dx
img = np.array(image)
img = img / 255.0
img = img.astype(np.float64)
x_dx = get_x_dx(img)
plt.imshow(x_dx, cmap='gray')
plt.title("x_dx")
plt.axis("off")
plt.show()
