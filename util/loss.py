import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



class Loss: 
    
    def tv(self, image):
        '''
        Total variation of image. L-1 norm of first derivative of image.
        First derivative of image is calucalted by forward difference.
        :param image: input image
        :return: total variation value of image
        '''
        x_loss = torch.mean((torch.abs(image[:,:,1:,:] - image[:,:,:-1,:])))
        y_loss = torch.mean((torch.abs(image[:,:,:,1:] - image[:,:,:,:-1])))

        return (x_loss + y_loss)
    
    def laplace(self, image):
        '''
        Laplace calculation of image. Laplace is second derivative of image.
        Second derivative of the image is calculated by 'forward difference - backward difference'
        :param image: input image
        :return: mean value of Laplace of image
        '''
        x_forward = image[:,:,1:,:] - image[:,:,:-1,:]
        x_backward = image[:,:,:-1,:] - image[:,:,1:,:]
        x_loss = torch.mean(torch.abs(x_forward - x_backward))

        y_forward = image[:,:,:,1:] - image[:,:,:,:-1]
        y_backward = image[:,:,:,:-1] - image[:,:,:,1:]
        y_loss = torch.mean(torch.abs(y_forward - y_backward))

        return (x_loss + y_loss)
    
    def iso(self, image):
        '''
        Loss for isotropic diffusion's regularization.
        :param image: input image
        :return: L2 norm of the delta image.
        '''
        x_loss = torch.mean(torch.norm((image[:,:,1:,:] - image[:,:,:-1,:]), p=2))
        y_loss = torch.mean(torch.norm((image[:,:,:-1,:] - image[:,:,1:,:]), p=2))

        return x_loss + y_loss
    
