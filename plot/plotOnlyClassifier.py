import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import sys
from util.utils import get_image, get_threshold_mask


class plot:
    def __init__(self, train_loader, val_loader, encoder, maskDecoder, device, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.maskDecoder = maskDecoder
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

    def visualize_stn(self, loader):
        with torch.no_grad():
            data,target,_ = next(iter(loader))
            data, target = data.to(self.device), target.to(self.device)
            data = data[:16]
            target = target[:16]
            input_tensor = data.cpu()
            target = target.cpu()

            mask = get_image(data, self.encoder, self.maskDecoder)
            threshold_mask = get_threshold_mask(mask)

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(input_tensor, nrow=4), True)
            
            mask_grid = self.convert_image_np(
                torchvision.utils.make_grid(mask, nrow=4), False)
            
            fore_grid = in_grid * mask_grid
            back_grid = in_grid * (1-mask_grid)

            threshold_mask_grid = self.convert_image_np(
                torchvision.utils.make_grid(threshold_mask, nrow=4), False)

            threshold_fore_grid = in_grid * threshold_mask_grid
            threshold_back_grid = in_grid * (1 - threshold_mask_grid)

            target_grid = self.convert_image_np(
                torchvision.utils.make_grid(target, nrow=4), False)
            target_fore_grid = in_grid * target_grid
            target_back_grid = in_grid * (1-target_grid)

            plt.close('all')
            fig = plt.figure(figsize=(16,12))
            fig.tight_layout()
            ax1 = fig.add_subplot(4,3,2)

            ax2 = fig.add_subplot(4,3,4)
            ax3 = fig.add_subplot(4,3,5)
            ax4 = fig.add_subplot(4,3,6)

            ax5 = fig.add_subplot(4,3,7)
            ax6 = fig.add_subplot(4,3,8)
            ax7 = fig.add_subplot(4,3,9)

            ax8 = fig.add_subplot(4,3,10)
            ax9 = fig.add_subplot(4,3,11)
            ax10 = fig.add_subplot(4,3,12)

            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)
            ax4.axes.get_xaxis().set_visible(False)
            ax4.axes.get_yaxis().set_visible(False)
            ax5.axes.get_xaxis().set_visible(False)
            ax5.axes.get_yaxis().set_visible(False)
            ax6.axes.get_xaxis().set_visible(False)
            ax6.axes.get_yaxis().set_visible(False)
            ax7.axes.get_xaxis().set_visible(False)
            ax7.axes.get_yaxis().set_visible(False)
            ax8.axes.get_xaxis().set_visible(False)
            ax8.axes.get_yaxis().set_visible(False)
            ax9.axes.get_xaxis().set_visible(False)
            ax9.axes.get_yaxis().set_visible(False)
            ax10.axes.get_xaxis().set_visible(False)
            ax10.axes.get_yaxis().set_visible(False)

            ax1.imshow(in_grid)
            ax1.set_title('original')

            ax2.imshow(target_grid, cmap='gray', vmin=0, vmax=1)
            ax2.set_title('target mask')
            ax3.imshow(target_fore_grid)
            ax3.set_title('target foreground')
            ax4.imshow(target_back_grid)
            ax4.set_title('target background')

            ax5.imshow(threshold_mask_grid, cmap='gray', vmin=0, vmax=1)
            ax5.set_title('thresholded mask')
            ax6.imshow(threshold_fore_grid)
            ax6.set_title('thresholded foreground')
            ax7.imshow(threshold_back_grid)
            ax7.set_title('thresholded background')

            maskPic = ax8.imshow(mask_grid, cmap='gray', vmin=0, vmax=1)
            divider = make_axes_locatable(ax8)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(maskPic, cax=cax, orientation='vertical')
            ax8.set_title('mask')
            ax9.imshow(fore_grid)
            ax9.set_title('foreground')
            ax10.imshow(back_grid)
            ax10.set_title('background')

            # plt.title('{}'.format(count))
            plt.tight_layout()     

            return fig

    def visualize_loss(self, trainResult, valResult):
        plt.clf()
        figure, axarr = plt.subplots(1, 4, figsize=(32,8))

        axarr[0].plot(trainResult[0], 'r-', label='train loss')
        axarr[0].fill_between(range(len(trainResult[0])), np.array(trainResult[0])-np.array(trainResult[1]), np.array(trainResult[0])+np.array(trainResult[1]),alpha=.1, color='r')
        axarr[0].plot(valResult[0], 'b-', label='val loss')
        axarr[0].fill_between(range(len(valResult[0])), np.array(valResult[0])-np.array(valResult[1]), np.array(valResult[0])+np.array(valResult[1]),alpha=.1, color='b')
        axarr[0].legend(loc='upper left')
        axarr[0].set_title('total loss and accuracy')

        axarr[1].plot(trainResult[4], 'r-', label='train foreground class loss')
        axarr[1].plot(valResult[4], 'g-', label='val foreground class loss')
        twin = axarr[1].twinx()
        twin.plot(trainResult[5], 'b-', label='train background class loss')
        twin.plot(valResult[5], 'c-', label='val background class loss')
        twin.legend(loc='upper left')
        axarr[1].set_title('class loss')
        axarr[1].legend(loc='upper right')
        
        axarr[2].plot(trainResult[2], 'r-', label='train mask region reg')
        axarr[2].plot(trainResult[3], 'g-', label='train mask smooth reg')
        axarr[2].plot(valResult[2], 'b-', label='val mask region reg')
        axarr[2].plot(valResult[3], 'c-', label='val mask smooth reg')
        axarr[2].set_title('regular')
        axarr[2].legend(loc='upper left')

        axarr[3].plot(trainResult[6], 'r-', label='train acc')
        axarr[3].plot(valResult[6], 'b-', label='val acc')
        axarr[3].set_title('accuracy')
        axarr[3].legend(loc='upper left')

        plt.tight_layout()    

        return figure

    def plotResult(self, epoch, trainResult, valResult):
        path = os.path.join('../../../result/mask', self.config.title, str(self.config.mr), str(self.config.ms))

        if epoch != 0: 
            # visualize train data
            trainPicPath = os.path.join(path, 'pic/train')
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/train')
            trainPic2 = self.visualize_stn(self.val_loader)
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
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/val')
            trainPic2 = self.visualize_stn(self.val_loader)
            try:
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(valPicPath)
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))

            