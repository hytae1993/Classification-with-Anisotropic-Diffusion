from __future__ import print_function

import os
import math
from numpy.lib.function_base import select

import torch
import torch.backends.cudnn as cudnn
from  torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from model.Vgg import Encoder, Decoder
from model.vggClassifier import vgg19_bn, vgg19, vgg16_bn, vgg16
from model.resNetClassifier import ResNet18, ResNet34
from util.loss import Loss
from util.progress_bar import progress_bar
from util.scheduler_learning_rate import *
from util.gaussian_smoothing import *
from util.utils import *

import numpy as np

from plot.plotClassification import classificationPlot

class classification(object):
    def __init__(self, config, training_loader, val_loader, diffusion):
        super(classification, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.seed = config.seed
        self.nEpochs = config.epoch
        self.lr = config.lr
        self.loss = Loss()
        self.diffusion = diffusion
        self.log_interval = config.log
        self.config = config
        
        self.train_loader = training_loader
        self.val_loader = val_loader

        self.encoder = None
        self.decoder = None
        self.classifier = None

        self.plot = None

        self.optimizer = {}

        self.crossCriterion = None

        self.train_loss = []
        self.val_loss = []

    def build_model(self):
        path = os.path.join('./savedModel/addSkip', self.config.diffusion+ '/diffusion_coefficient_' + str(self.config.diffusionCoeff))
        if self.config.diffusion == 'anisotropic':
            self.encoder = Encoder()
            self.decoder = Decoder(last=3, skip=True, concat=False)
            self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder')))
            self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder')))

            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

        # classifier
        if self.config.classifier == 'vgg16':
            self.classifier = vgg16(pretrained=False, num_classes=self.config.num_classes)
        elif self.config.classifier == 'vgg16_bn':
            self.classifier = vgg16_bn(pretrained=False, num_classes=self.config.num_classes)
        elif self.config.classifier == 'vgg19':
            self.classifier = vgg16_bn(pretrained=False, num_classes=self.config.num_classes)
        elif self.config.classifier == 'vgg19_bn':
            self.classifier = vgg16_bn(pretrained=False, num_classes=self.config.num_classes)
        elif self.config.classifier == 'ResNet18':
            self.classifier = ResNet18(num_classes=self.config.num_classes)
        elif self.config.classifier == 'ResNet34':
            self.classifier = ResNet34(num_classes=self.config.num_classes)
        self.classifier = self.classifier.to(self.device)

        self.plot = classificationPlot(self.train_loader, self.val_loader, self.encoder, self.decoder, self.classifier, self.device, self.config)
        self.crossCriterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer['classifier'] = torch.optim.SGD(self.classifier.parameters(), lr=self.lr, weight_decay=self.config.decay)

    def diffuseData(self, data, epoch):
        # if using diffused image with constant anisotropic diffused image
        if self.config.diffusion == 'anisotropic':
            encoderBlock, bottleNeck = self.encoder(data)
            diffusedImage, _ = self.decoder(blocks=encoderBlock, bottleNeck=bottleNeck)  
        
        elif self.config.diffusion == 'isotropic':
            kernel = gaussian_Smoothing(self.config.kernel_size, self.config.std)
            diffusedImage = kernel.smoothing(data, self.config.kernel_size, self.config.std)

        # if using diffused image with annealing isotropic diffusion coefficient
        elif self.config.diffusion == 'annealing':
            with torch.no_grad():
                # The diffusedImage of the last epoch must be same as the input image
                if epoch == self.nEpochs:
                    diffusedImage = data
                else:
                    kernel = gaussian_Smoothing(self.config.kernel_size, self.config.std)
                    kernel_size, std = kernel.annealing(epoch, self.nEpochs)
                    diffusedImage = kernel.smoothing(data, kernel_size, std)

        # if using original image
        elif self.config.diffusion == 'original':
            diffusedImage = data

        return diffusedImage

    def run(self, epoch, data_loader, work):
        if work == 'train':
            self.classifier.train()
        elif work == 'val':
            self.classifier.eval()

        lossList = []
        acc = 0

        iter = 0
        num_data = 0

        for batch_num, (data, target) in enumerate(data_loader):
            iter += 1
            num_data += data.size(0)
            data = data.to(self.device)
            target = target.to(self.device)

            if work == 'train':
                diffusedImage = self.diffuseData(data, epoch)

                classScore, _ = self.classifier(diffusedImage)

                loss = self.crossCriterion(classScore, target)

                self.optimizer['classifier'].zero_grad()
                loss.backward()
                self.optimizer['classifier'].step()

            elif work == 'val':
                with torch.no_grad():
                    if self.config.diffusion == 'annealing':
                        diffusedImage = data
                    else:
                        diffusedImage = self.diffuseData(data, epoch)

                    classScore, _ = self.classifier(diffusedImage)

                    loss = self.crossCriterion(classScore, target)

            lossList.append(loss.item())

            pred = classScore.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc += correct

            progress_bar(batch_num, len(data_loader))

        return np.mean(lossList), np.std(lossList), 100.*acc/num_data
        
    def runner(self):

        for i in range(3):
            self.train_loss.append([])
            self.val_loss.append([])
       
        self.build_model()

        self.plot.plotResult(epoch=0, trainResult=None, valResult=None)
        
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))

            trainResult = self.run(epoch, self.train_loader, 'train')
            valResult = self.run(epoch, self.val_loader, 'val')

            for i in range(3):
                self.train_loss[i].append(trainResult[i])
                self.val_loss[i].append(valResult[i])

            if epoch % self.log_interval == 0 or epoch == 1:
                self.plot.plotResult(epoch, self.train_loss, self.val_loss)

            if epoch == self.nEpochs:
                self.plot.plotResult(epoch, self.train_loss, self.val_loss)
                # save_diffusion_model(self.encoder, self.decoder, self.config)
                saveExcelClass(self.train_loss, self.val_loss, self.config)

