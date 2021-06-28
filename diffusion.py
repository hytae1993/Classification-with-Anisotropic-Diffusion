from __future__ import print_function

import os
import math

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from model.Vgg import Encoder, Decoder
from util.loss import Loss
from util.progress_bar import progress_bar
from util.scheduler_learning_rate import *
from util.utils import *

import numpy as np

from plot.plotDiffusion import diffusionPlot

class diffusor(object):
    def __init__(self, config, training_loader, val_loader, diffusion):
        super(diffusor, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.seed = config.seed
        self.nEpochs = config.epoch
        self.lr = config.lr
        self.diffusionCoeff = config.diffusionCoeff
        self.loss = Loss()
        self.diffusion = diffusion
        self.log_interval = config.log
        self.config = config
        
        self.train_loader = training_loader
        self.val_loader = val_loader

        self.encoder = None
        self.decoder = None

        self.plot = None

        self.optimizer = {}

        self.mseCriterion = None
        self.crossCriterion = None
        self.l1Criterion = None

        self.train_loss = []
        self.val_loss = []

        self.train_image = []
        self.val_image = []


    def build_model(self):
        self.encoder = Encoder()
        self.decoder = Decoder(last=3, skip=True, concat=False)     # concat=False면 add connection, skip=False면 no skip 

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.plot = diffusionPlot(self.train_loader, self.val_loader, self.encoder, self.decoder, self.device, self.config)

        self.mseCriterion = torch.nn.MSELoss()
        self.l1Criterion = torch.nn.L1Loss()

        if self.CUDA:
            cudnn.benchmark = True
            self.mseCriterion.cuda()
            self.l1Criterion.cuda()

        self.optimizer['encoder'] = torch.optim.SGD(self.encoder.parameters(), lr=self.lr, weight_decay=1e-4)
        self.optimizer['decoder'] = torch.optim.SGD(self.decoder.parameters(), lr=self.lr, weight_decay=1e-4)


    def run(self, epoch, data_loader, work):
        if work == 'train':
            self.encoder.train()
            self.decoder.train()
        elif work == 'val':
            self.encoder.eval()
            self.decoder.eval()

        lossList = []
        Regular = 0
        dataFidelity = 0

        iter = 0
        num_data = 0

        for batch_num, (data, target) in enumerate(data_loader):
            iter += 1
            num_data += data.size(0)
            data = data.to(self.device)

            if work == 'train':

                encoderBlock, bottleNeck = self.encoder(data)
                diffusedImage, decoderBlock = self.decoder(blocks=encoderBlock, bottleNeck=bottleNeck)   

                if self.diffusion == 'anisotropic':
                    regularization = self.loss.tv(diffusedImage)
                elif self.diffusion == 'isotropic':
                    regularization = self.loss.iso(diffusedImage)

                fidelity = self.mseCriterion(data, diffusedImage)
                regularization = self.diffusionCoeff * regularization

                loss = fidelity + regularization

                self.optimizer['encoder'].zero_grad()
                self.optimizer['decoder'].zero_grad()
                loss.backward()
                self.optimizer['encoder'].step()
                self.optimizer['decoder'].step()

            elif work == 'val':
                with torch.no_grad():
                    encoderBlock, bottleNeck = self.encoder(data)
                    diffusedImage, decoderBlock = self.decoder(blocks=encoderBlock, bottleNeck=bottleNeck)   

                    if self.diffusion == 'anisotropic':
                        regularization = self.loss.tv(diffusedImage)
                    elif self.diffusion == 'isotropic':
                        regularization = self.loss.iso(diffusedImage)

                    fidelity = self.mseCriterion(data, diffusedImage)
                    regularization = self.diffusionCoeff * regularization

                    loss = fidelity + regularization

            Regular += (regularization.item() * data.size(0))
            dataFidelity += (fidelity.item() * data.size(0))
            lossList.append(loss.item())

            progress_bar(batch_num, len(data_loader))

        return np.mean(lossList), np.std(lossList), dataFidelity/num_data, Regular/num_data
                    
        
    def runner(self):

        for i in range(4):
            self.train_loss.append([])
            self.val_loss.append([])
       
        self.build_model()
        
        # visualize initialize data
        self.plot.plotResult(epoch=0, trainResult=None, valResult=None)

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))

            trainResult = self.run(epoch, self.train_loader, 'train')
            valResult = self.run(epoch, self.val_loader, 'val')

            for i in range(4):
                self.train_loss[i].append(trainResult[i])
                self.val_loss[i].append(valResult[i])

            if epoch % self.log_interval == 0 or epoch == 1:
                self.plot.plotResult(epoch, self.train_loss, self.val_loss)

            if epoch == self.nEpochs:
                self.plot.plotResult(epoch, self.train_loss, self.val_loss)
                save_diffusion_model(self.encoder, self.decoder, self.config)

