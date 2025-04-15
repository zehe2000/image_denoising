import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math


def resize_image_to_power_of_2(image: Image.Image) -> Image.Image:
    """
    Resizes the image so its width is a power of 2.
    """
    width, height = image.size
    if width > 0 and (width & (width - 1)) == 0:
        print(f"The image width ({width}) is already a power of 2.")
        return image
    else:
        new_width = 2 ** int(math.floor(math.log2(width)))
        print(f"Resizing image from width {width} to {new_width}.")
        resized_image = image.resize((new_width, height), resample=Image.Resampling.LANCZOS)
        return resized_image

def sign(a: float) -> float:
    """
    Return the sign of a number.
    """
    if a > 0.0:
        return 1.0
    elif a < 0.0:
        return -1.0
    else:
        return 0.0