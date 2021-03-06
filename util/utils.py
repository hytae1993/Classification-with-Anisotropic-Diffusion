import torch
import numpy as np
import os
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt
import skimage.transform
from skimage.util.dtype import img_as_float
import pandas as pd

from util.gaussian_smoothing import *

def get_diffused_image(input, encoder, decoder, config, epoch=None):
    """
    Get the mask
    :param input: input image
    :param encoder: encoder
    :param decoder: decoder
    :return image: diffused image
    """
    with torch.no_grad():
        if config.diffusion == 'anisotropic':
            block, bottleNeck = encoder(input)  # autoencoder
            image,_ = decoder(block, bottleNeck)
        elif config.diffusion == 'isotropic' and config.model == 'diffusion':
            block, bottleNeck = encoder(input)  # autoencoder
            image,_ = decoder(block, bottleNeck)
        elif config.diffusion == 'isotropic' and config.model == 'classification':
            kernel = gaussian_Smoothing(config.kernel_size, config.std)
            image = kernel.smoothing(input, config.kernel_size, config.std)
        elif config.diffusion == 'annealing':
            if epoch == config.epoch:
                image = input
            else:
                kernel = gaussian_Smoothing(config.kernel_size, config.std)
                kernel_size, std = kernel.annealing(epoch, config.epoch)
                image = kernel.smoothing(input, kernel_size, std)
        elif config.diffusion == 'original':
            image = input

        image = image.cpu()

        return image

def get_cam(input, target, classifier):
    """
    Get the class activation map
    :return image: cam about ground truth label and predcit label
    """
    with torch.no_grad():
        # classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        params = list(classifier.parameters())

        outputs, feature = classifier(input)
        _, predicted = torch.max(outputs, 1)

        predict_cam = list()
        target_cam = list()
        for i in range(len(input)):
            overlay = params[-2][int(predicted[i])].matmul(feature[i].reshape(512,49)).reshape(7,7).cpu().data.numpy()
            overlay = overlay.astype('float64')
            overlay = overlay - np.min(overlay)
            overlay = overlay / (np.max(overlay) - np.min(overlay))
            overlay = skimage.transform.resize(overlay, [224,224])
            overlay = FT.to_tensor(overlay)
            predict_cam.append(overlay)

            overlay = params[-2][int(target[i])].matmul(feature[i].reshape(512,49)).reshape(7,7).cpu().data.numpy()
            overlay = overlay - np.min(overlay)
            overlay = overlay / (np.max(overlay) - np.min(overlay))
            overlay = skimage.transform.resize(overlay, [224,224])
            overlay = FT.to_tensor(overlay)
            target_cam.append(overlay)

        predict_cam = torch.stack(predict_cam, dim=0)
        target_cam = torch.stack(target_cam, dim=0)
    return predict_cam, target_cam

def save_diffusion_model(encoder, decoder, config):
    path = './savedModel/concatSkip'
    aniso_path = os.path.join(path, config.diffusion + '/diffusion_coefficient_' + str(config.diffusionCoeff))
    try: 
        torch.save(encoder.state_dict(), os.path.join(aniso_path, 'encoder'))
        torch.save(decoder.state_dict(), os.path.join(aniso_path, 'decoder'))
    except FileNotFoundError:
        os.makedirs(aniso_path)
        torch.save(encoder.state_dict(), os.path.join(aniso_path, 'encoder'))
        torch.save(decoder.state_dict(), os.path.join(aniso_path, 'decoder'))

def saveExcelClass(trainResult, valResult, config):
    trainResult = np.transpose(trainResult)
    train_df = pd.DataFrame(trainResult, columns=['total loss mean', 'total loss std', 'accuracy'])

    valResult = np.transpose(valResult)
    val_df = pd.DataFrame(valResult, columns=['total loss mean', 'total loss std', 'accuracy'])

    path = './excel/'
    if config.diffusion != 'original':
        path = os.path.join(path, config.diffusion + '/diffusion_coefficient_' + str(config.diffusionCoeff))
    else:
        path = os.path.join(path, config.diffusion)

    try:
        train_df.to_csv(os.path.join(path + '/trainLoss.csv'))
        val_df.to_csv(os.path.join(path + 'valLoss.csv'))
    except FileNotFoundError:
        os.makedirs(path)
        train_df.to_csv(os.path.join(path + '/trainLoss.csv'))
        val_df.to_csv(os.path.join(path + '/valLoss.csv'))