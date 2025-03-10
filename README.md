# Optimization Algorithms for Image-Based Energy Minimization

This project implements several optimization algorithms designed to minimize an energy function used in image processing. The energy function is defined as

![Energy Function](energy_function.png)


where \(y\) is the observed (noisy) image and \(x\) is the image being optimized. The project provides implementations of:

- **Gradient Descent**  
- **BFGS** (a quasi-Newton method)  
- **Gauss–Newton**

These methods can be applied to problems such as denoising or deblurring images by minimizing the above energy function.

---

## Project Structure

The repository is organized into two main directories to separate the optimization methods based on their order:

- **`gd_and_bfgs/`**  
  Contains the implementations of:
  - **Gradient Descent**
  - **BFGS**

- **`gauss_newton/`**  
  Contains the implementation of the **Gauss–Newton** method.

Other directories and files include:


- **`requirements.txt`**  
  A list of Python packages required to run the project (e.g., numpy, matplotlib, imageio).

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/optimization-project.git
   cd optimization-project
