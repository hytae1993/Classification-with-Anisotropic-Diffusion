from __future__ import print_function

import argparse
import os

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torch
import numpy as np

import random

from diffusion import diffusor
from classification import classification

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch classification')
# hyper-parameters
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log', type=int, default=1)
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight Decay. Default=0.0001')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--model', '-m', type=str, default='diffusion', help='choose which model is going to use')
parser.add_argument('--size', type=int, default=224, help='image size for resize')
parser.add_argument('--diffusion', type=str, default='anisotropic', help='which diffusion')
parser.add_argument('--diffusionCoeff', '--dc', type=float, default=0.001, help='diffusion coefficient')
parser.add_argument('--classifier', type=str, default='ResNet18', help='which classifier')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--title', type=str, default='anisotropic_diffusion')

args = parser.parse_args()

class main:
    def __init__(self):
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def dataLoad(self):
        # ===========================================================
        # Set train dataset & test dataset
        # ===========================================================
        print('===> Loading datasets')

        def seed_worker(self):
            np.random.seed(args.seed)
            random.seed(args.seed)
        
        transform = transforms.Compose([
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        trainset = torchvision.datasets.STL10(root='../../../dataset/stl10', split='train', download=True, transform=transform)
        testset = torchvision.datasets.STL10(root='../../../dataset/stl10', split='test', download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=2)

    def modelCall(self):
        
        if args.model == 'diffusion':          # anisotropic and isotropic diffusion auto-encoder
            self.model = diffusor(args, self.train_loader, self.val_loader, args.diffusion)

        elif args.model == 'classification':   # classification
            self.model = classification(args, self.train_loader, self.val_loader, args.diffusion)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

    main = main()
    main.dataLoad()
    main.modelCall()
    
    main.model.runner()