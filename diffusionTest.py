import torch 
from util.gaussian_smoothing import *
import PIL
from model.Vgg import Encoder, Decoder
import os
import matplotlib.pyplot as plt

def gaussianBlur():
    path = './exampleImage/original.png'
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

    plt.savefig('./exampleImage/gaussian blurred_{}.png'.format(kernel_size))

def anisotropicBlur(diffusionCoefficient):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imagePath = './exampleImage/original.png'
    image = PIL.Image.open(imagePath)
    
    data = transforms.Resize((224,224))(image)
    data = transforms.ToTensor()(data)
    data = data.unsqueeze(dim=0).float()
    # data.to(device=device)
    
    modelPath = os.path.join('./savedModel/addSkip', 'anisotropic'+ '/diffusion_coefficient_' + str(diffusionCoefficient))
    encoder = Encoder()
    decoder = Decoder(last=3, skip=True, concat=False)
    encoder.load_state_dict(torch.load(os.path.join(modelPath, 'encoder')))
    decoder.load_state_dict(torch.load(os.path.join(modelPath, 'decoder')))
    # encoder.to(device=device)
    # decoder.to(device=device)

    encoderBlock, bottleNeck = encoder(data)
    diffusedImage, _ =  decoder(blocks=encoderBlock, bottleNeck=bottleNeck)  

    img = data[0].numpy().transpose((1, 2, 0))
    diffusedImage = diffusedImage[0].detach().numpy().transpose((1, 2, 0))

    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    # axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('image source')
    axs[0].imshow(img)

    axs[1].axis('off')
    axs[1].set_title('image blurred')
    axs[1].imshow(diffusedImage)

    plt.savefig('./exampleImage/anisotropic blurred_{}.png'.format(diffusionCoefficient))

if __name__ == "__main__":
    # gaussianBlur()
    anisotropicBlur(0.01)