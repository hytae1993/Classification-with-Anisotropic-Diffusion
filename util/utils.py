import torch
import numpy as np
import torch.nn as nn

def get_image(input, encoder, maskDecoder):
    """
    Get the mask
    :param input: input image
    :param encoder: encoder
    :param maskDecoder: decoder
    :return mask: the mask which bi-partition the input image to the foregroun and the background
    """
    with torch.no_grad():
        block, bottleNeck = encoder(input)  # autoencoder
        mask,_ = maskDecoder(block, bottleNeck)
        mask = mask.cpu()

        return mask

def get_threshold_mask(mask):
    """
    Threshold the mask based on 0.5.
    :param mask: input mask
    :return: thresholded mask
    """
    with torch.no_grad():
        out = (mask>0.5).float()
    
    return out