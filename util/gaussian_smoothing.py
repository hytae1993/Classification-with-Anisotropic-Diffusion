import torch
import kornia
import numpy as np
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class gaussian_Smoothing():
    def __init__(self, kernel_size, std):
        # set gaussian kernel
        # you need to set kernel size and standard deviation
        self.kernel_size = kernel_size
        self.std = std

    def smoothing(self, image, kernel_size, std):
        '''
        :gauss: make gaussian smoothing kernel and apply the kernel to the input image. Need kernel size and standard deviation.
        :return: gaussian smoothed image
        '''
        # kernel = kornia.filters.get_gaussian_kernel2d(self.kernel_size, self.std) # made kernel
        gauss = kornia.filters.GaussianBlur2d((kernel_size, kernel_size), (std, std))
        blur = gauss(image)

        return blur

    def annealing(self, epoch, totalEpoch):
        '''
        : y = 'first value - 2*epoch' for the annealing function for the example. You need to change the function properly.
        : You can use the totalEpoch for annealig function. See scheduling_learning_rate.py
        : The image of the last epoch must be the original image.
        '''
        kernel_size = self.kernel_size - 2 * epoch 
        std = self.std - epoch

        return kernel_size, std


if __name__ == "__main__":
    kernel_size = (25,25)
    std = (100, 100)
    test = gaussian_Smoothing(kernel_size, std)

    # set your testing image directory
    path = '../../../../dataset/dogcat/train/cat/cat.3.jpg'
    image = PIL.Image.open(path)
    data = transforms.ToTensor()(image)
    data = data.unsqueeze(dim=0)

    blur = test.smoothing(data, kernel_size, std)

    img = data[0].numpy().transpose((1, 2, 0))
    blur = blur[0].numpy().transpose((1, 2, 0))

    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('image source')
    axs[0].imshow(img)

    axs[1].axis('off')
    axs[1].set_title('image blurred')
    axs[1].imshow(blur)

    plt.savefig('gaussian blurred.png')

    # print(kernel)