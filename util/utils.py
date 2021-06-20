import torch
import numpy as np
import os
import torchvision.transforms.functional as FT
import kornia
import matplotlib.pyplot as plt
import skimage.transform

def get_diffused_image(input, encoder, decoder, config=None, epoch=None):
    """
    Get the mask
    :param input: input image
    :param encoder: encoder
    :param decoder: decoder
    :return image: diffused image
    """
    with torch.no_grad():
        if encoder != None and decoder != None:
            block, bottleNeck = encoder(input)  # autoencoder
            image,_ = decoder(block, bottleNeck)
        else:
            if config.diffusion == 'annealing':
                image = kornia.gaussian_blur2d(input, kernel_size=(3,3), sigma=(1,1))
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
    path = './savedModel/'
    aniso_path = os.path.join(path, config.diffusion + '/diffusion_coefficient_' + str(config.diffusionCoeff))
    try: 
        torch.save(encoder.state_dict(), os.path.join(aniso_path, 'encoder'))
        torch.save(decoder.state_dict(), os.path.join(aniso_path, 'decoder'))
    except FileNotFoundError:
        os.makedirs(aniso_path)
        torch.save(encoder.state_dict(), os.path.join(aniso_path, 'encoder'))
        torch.save(decoder.state_dict(), os.path.join(aniso_path, 'decoder'))