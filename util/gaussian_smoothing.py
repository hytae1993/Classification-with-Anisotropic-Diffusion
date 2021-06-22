import torch
import kornia
import numpy as np
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class smoothing():
    def __init__(self, kernel_size, std):
        # set gaussian kernel
        # you need to set kernel size and standard deviation
        self.kernel_size = kernel_size
        self.std = std
        self.gauss = kornia.filters.GaussianBlur2d(kernel_size, std)

    def smoothing(self, image):

        blur = self.gauss(image)

        return blur

        

if __name__ == "__main__":
    kernel_size = (11,11)
    std = (10.5, 10.5)
    test = smoothing(kernel_size, std)

    # set your testing image directory
    path = '../../../../dataset/dogcat/train/cat/cat.3.jpg'
    image = PIL.Image.open(path)
    data = transforms.ToTensor()(image)
    data = data.unsqueeze(dim=0)

    blur = test.smoothing(data)

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