# Classification-with-Anisotropic-Diffusion
The change in classification with the change of the diffusion coefficient of anisotropic diffusion and isotropic diffusion.

Require torch, kornia

## Anisotropic Diffusion VS Isotropic Diffusion VS Original
### 1. Anisotropic Diffusion
1) Train Auto Encoder for Anisotropic Diffusion. (Total variation regularization)
2) Get the classification loss and classification score per diffusion coefficient.
3) Get the class activation map (CAM or Grad-CAM) per diffusion coefficient.
### 2. Isotropic Diffusion
1) Train Auto Encoder for Isotropic Diffusion. (Laplace regularization)
2) Get the classification loss and classification score per diffusion coefficient.
3) Get the class activation map (CAM or Grad-CAM) per diffusion coefficient.
### 3. Original
1) Using Original Input Image.
2) Get the classification loss and classification score.
3) Get the class activation map (CAM or Grad-CAM).
### 4. Annealing diffusion coefficient
1) Not using Auto Encoder. Use some kernels.
2) The kernels are Gaussian Kernel.
3) The Kernels change every epoch. 
4) The kernels' std become smaller as learning progresses. (The kernels' size become smaller.)

## Diffusing process
1) For constant diffusion coefficient of anisotropic diffsuion and isotropic diffusion, use auto-encoder based on VGG-19.
2) For annealing constant diffusion coefficient, use gaussian kernel. It is applied by convolution filter. So we do not use neural network.
### 1. Anisotropic Diffusion
Run the main.py with
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=diffusion --diffusion=anisotropic --dc=0.001
```
### 2. Isotropic Diffusion
Run the main.py with
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=diffusion --diffusion=isotropic --dc=0.001
```
## Classification procee
For classifier, we use ```vgg16, vgg16_bn, vgg19, vgg19_bn, ResNet18, ResNet34```. ```bn means batch normalization```

Specify the classifier with ``` --classifier=vgg16 ```.
### 1. Classify the anisotropic diffused image
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=anisotropic --classifier=vgg16 --num_classes=10
```
### 2. Classify the isotropic diffused image
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=isotropic --classifier=vgg16 --num_classes=10
```
### 3. Classify the diffused image with annealing diffusion coefficient
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=annealing --classifier=vgg16 --num_classes=10
```
### 4. Classify the original image
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=original --classifier=vgg16 --num_classes=10
```
## Some Results
### Cam of the original image
![image](https://user-images.githubusercontent.com/32087995/122704440-b479c400-d28e-11eb-8fa3-e806981afd6e.png)
