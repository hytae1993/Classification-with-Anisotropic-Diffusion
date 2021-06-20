import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import sys
from util.utils import get_cam, get_diffused_image

class classificationPlot:
    def __init__(self, train_loader, val_loader, encoder, decoder, classifier, device, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.device = device
        self.config = config

    def convert_image_np(self, inp, image):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array((0.485, 0.456, 0.406))
        std = np.array((0.229, 0.224, 0.225))
        if image:
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

        return inp

    def visualize_stn(self, loader, epoch):
        with torch.no_grad():
            data,target = next(iter(loader))
            data = data.to(self.device)
            data = data[:16]
            input_tensor = data.cpu()

            diffusedImage = get_diffused_image(data, self.encoder, self.decoder, self.config, epoch)
            predictCamImage, targetCamImage = get_cam(data, target, self.classifier)

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(input_tensor, nrow=4), True)
            
            diffused_grid = self.convert_image_np(
                torchvision.utils.make_grid(diffusedImage, nrow=4), True)

            predict_cam_grid = self.convert_image_np(
                torchvision.utils.make_grid(predictCamImage, nrow=4), False)
            predict_cam_grid = predict_cam_grid[:,:,0]
            predict_cam_grid = np.squeeze(predict_cam_grid)

            target_cam_grid = self.convert_image_np(
                torchvision.utils.make_grid(targetCamImage, nrow=4), False)
            target_cam_grid = target_cam_grid[:,:,0]
            target_cam_grid = np.squeeze(target_cam_grid)
            
            plt.close('all')
            fig = plt.figure(figsize=(32,20))
            fig.tight_layout()
            ax1 = fig.add_subplot(2,2,1)
            ax2 = fig.add_subplot(2,2,2)
            ax3 = fig.add_subplot(2,2,3)
            ax4 = fig.add_subplot(2,2,4)

            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)
            ax4.axes.get_xaxis().set_visible(False)
            ax4.axes.get_yaxis().set_visible(False)

            ax1.imshow(in_grid)
            ax1.set_title('original')

            ax2.imshow(diffused_grid)
            ax2.set_title('diffused')

            ax3.imshow(in_grid)
            ax3.imshow(predict_cam_grid, cmap='jet', alpha=0.4, interpolation='nearest')
            ax3.set_title('predict cam')

            ax4.imshow(in_grid)
            ax4.imshow(target_cam_grid, cmap='jet', alpha=0.4, interpolation='nearest')
            ax4.set_title('target cam')

            # plt.title('{}'.format(count))
            plt.tight_layout()     

            return fig

    def visualize_loss(self, trainResult, valResult):
        plt.clf()
        figure, axarr = plt.subplots(1, 2, figsize=(18,8))

        axarr[0].plot(trainResult[0], 'r-', label='train loss')
        axarr[0].fill_between(range(len(trainResult[0])), np.array(trainResult[0])-np.array(trainResult[1]), np.array(trainResult[0])+np.array(trainResult[1]),alpha=.1, color='r')
        axarr[0].plot(valResult[0], 'b-', label='val loss')
        axarr[0].fill_between(range(len(valResult[0])), np.array(valResult[0])-np.array(valResult[1]), np.array(valResult[0])+np.array(valResult[1]),alpha=.1, color='b')
        axarr[0].legend(loc='upper left')
        axarr[0].set_title('total loss')

        axarr[1].plot(trainResult[2], 'r-', label='train accuracy')
        axarr[1].plot(valResult[2], 'g-', label='val accuracy')
        axarr[1].set_title('accuracy')
        axarr[1].legend(loc='upper left')
        
        plt.tight_layout()    

        return figure

    def plotResult(self, epoch, trainResult, valResult):
        if self.config.diffusion == 'anisotropic' or self.config.diffusion == 'isotropic':
            path = os.path.join('../../../result/classification', self.config.diffusion, self.config.classifier + '/diffusion_coefficient_' + str(self.config.diffusionCoeff))
        elif self.config.diffusion == 'annealing' or self.config.diffusion == 'original':
            path = os.path.join('../../../result/classification', self.config.diffusion, self.config.classifier)

        if epoch != 0: 
            # visualize train data
            trainPicPath = os.path.join(path, 'pic/train')
            trainPic1 = self.visualize_stn(self.train_loader, epoch)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/val')
            trainPic2 = self.visualize_stn(self.val_loader, epoch)
            try:
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(valPicPath)
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))

            # visualize loss graph
            lossPath = os.path.join(path, 'graph')
            loss = self.visualize_loss(trainResult, valResult)
            try:
                loss.savefig(os.path.join(lossPath + '/loss.png'))
            except FileNotFoundError:
                os.makedirs(lossPath)
                loss.savefig(os.path.join(lossPath + '/loss.png'))

        elif epoch == 0:
             # visualize train data
            trainPicPath = os.path.join(path, 'pic/train')
            trainPic1 = self.visualize_stn(self.train_loader, epoch)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/val')
            trainPic2 = self.visualize_stn(self.val_loader, epoch)
            try:
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(valPicPath)
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))

            