# Classification-with-Anisotropic-Diffusion
The change in classification with the change of the diffusion coefficient of anisotropic diffusion and isotropic diffusion.


### Requirements
Require torch, kornia, skimage

In Linux os 

pytorch        -> pip3 install torch torchvision 

kornia,skimage -> pip3 install kornia scikit-image

### Change repository to your own address

/main.py 

: line 63, 64 '../../../dataset/stl10' -> 'own address'    

/plot/plotClassification.py 

: line 116, 118 '../../../result/classification' -> 'own address'

/plot/plotDiffusion.py

: line 93 '../../../result/diffusion' -> 'own address'


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
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=anisotropic --classifier=vgg16 --num_classes=10 --diffusionCoeff=0.1
```
### 2. Classify the isotropic diffused image
```--kernel_size``` is for the gaussian kernel size of the gaussian smoothing.

```--std``` is for the standard deviation of the gaussain smoothing.
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=isotropic --kernel_size=7 --std=5 --classifier=vgg16 --num_classes=10
```
### 3. Classify the diffused image with annealing diffusion coefficient
```--kernel_size``` is for the gaussian kernel size of the gaussian smoothing at the first epoch.

```--std``` is for the standard deviation of the gaussain smoothing at the first epoch.

The last input image of training, is the original image. And, at inference time, the original image is used.
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=annealing --kernel_size=111 --std=60 --classifier=vgg16 --num_classes=10
```
### 4. Classify the original image
```
python3 main.py --epoch=100 --gpu=0 --batchSize=16 --lr=0.001 --model=classification --diffusion=original --classifier=vgg16 --num_classes=10 --decay=1e-4
```
## Some Results
### Anisotropic diffusioned image
![image](https://user-images.githubusercontent.com/32087995/123083340-ec356700-d45a-11eb-8018-69892caaee87.png)

### Isotropic diffusioned image
More blurred with big size kernel and large value of standard deviation.
![image](https://user-images.githubusercontent.com/32087995/123078854-58fa3280-d456-11eb-917f-69abb11ef4b1.png)


### Cam of the original images
![image](https://user-images.githubusercontent.com/32087995/122704440-b479c400-d28e-11eb-8fa3-e806981afd6e.png)


### Cam of the original images and diffused images
aniso_k: anisotropic diffusion with k-value of the diffusion coefficient
iso_m: isotropic diffusion with m-size of the kernel size for the gaussian noise
![image](https://user-images.githubusercontent.com/32087995/134683184-c2f40fb5-a1fe-42a7-b13a-0d91867a9b60.png)


### Classification accuracy of each diffusion process
![image](https://user-images.githubusercontent.com/32087995/134683250-f5d3b1e2-0b3f-43ed-ad65-df2cd13ae5de.png)
It means that anisotropic diffusion can get similar classification accuracy of original images, because anisotropic diffusion preserve the important information of the objects. But, isotropic diffusion cannot preserve the information of the objects, so the classification accuracies are lower than original images and aniostropic diffused images.


