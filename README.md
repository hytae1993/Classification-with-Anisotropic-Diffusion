# Classification-with-Anisotropic-Diffusion
The change in classification with the change of the diffusion coefficient of anisotropic diffusion and isotropic diffusion..

## Anisotropic Diffusion VS Isotropic Diffusion VS Original
### Anisotropic Diffusion
1. Train Auto Encoder for Anisotropic Diffusion. (Total variation regularization)
2. Get the classification loss and classification score per diffusion coefficient.
3. Get the class activation map (CAM or Grad-CAM) per diffusion coefficient.
### Isotropic Diffusion
1. Train Auto Encoder for Anisotropic Diffusion. (Laplace regularization)
2. Get the classification loss and classification score per diffusion coefficient.
3. Get the class activation map (CAM or Grad-CAM) per diffusion coefficient.
### Original
1. Using Original Input Image.
2. Get the classification loss and classification score.
3. Get the class activation map (CAM or Grad-CAM).
### Annealing diffusion coefficient
1. Not using Auto Encoder. Use some kernels.
2. The kernels are Gaussian Kernel.
3. The Kernels change every epoch. 
4. The kernels' std become smaller as learning progresses. (The kernels' size become smaller.)
