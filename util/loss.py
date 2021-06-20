import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



class Loss: 
    
    def __init__(self):
        self.area_loss_coef = 8
        self.smoothness_loss_coef = 0.5
        self.preserver_loss_coef = 0.3
        self.area_loss_power = 0.3
    
    def tv(self, image):
        x_loss = torch.mean((torch.abs(image[:,:,1:,:] - image[:,:,:-1,:])))
        y_loss = torch.mean((torch.abs(image[:,:,:,1:] - image[:,:,:,:-1])))

        return (x_loss + y_loss)

    def regionLoss(self, image):
        mask_mean = F.avg_pool2d(image, image.size(2), stride=1).squeeze().mean()

        return mask_mean

    
