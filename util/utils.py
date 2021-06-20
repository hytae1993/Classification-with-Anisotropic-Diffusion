import torch
import numpy as np
import os

def get_diffused_image(input, encoder, decoder):
    """
    Get the mask
    :param input: input image
    :param encoder: encoder
    :param decoder: decoder
    :return image: diffused image
    """
    with torch.no_grad():
        block, bottleNeck = encoder(input)  # autoencoder
        image,_ = decoder(block, bottleNeck)
        image = image.cpu()

        return image

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