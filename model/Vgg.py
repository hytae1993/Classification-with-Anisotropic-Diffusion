from numpy.core.shape_base import block
import torch
from torch import nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=5, wf=5, padding=True, batch_norm=True):
        super(Encoder, self).__init__()
        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(
                convBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.max_pool2d(x, 2)

        return blocks, x            

class convBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(convBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding), bias=False))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding), bias=False))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class Decoder(nn.Module):
    def __init__(self, depth=5, wf=5, padding=True, batch_norm=True, last=1, skip=True, concat=True):
        super(Decoder, self).__init__()
        self.depth = depth
        self.last_channel = last
        self.skip = skip
        prev_channels = 512

        self.up_path = nn.ModuleList()

        for i in reversed(range(depth-1)):     
            self.up_path.append(
                upsample(prev_channels, 2 ** (wf + i), padding, batch_norm, skip=skip, concat=concat)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(
            nn.Conv2d(prev_channels, self.last_channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
        
    def forward(self, blocks, bottleNeck):
        block = []
        x = bottleNeck

        for i, up in enumerate(self.up_path):
            if self.skip:
                x = up(x, blocks[-i-2])
                block.append(x)
            else:
                x = up(x, None)
                block.append(x)
        mask = self.last(x)
        return mask, block

class upsample(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, skip=True, concat=True):
        super(upsample, self).__init__()
        self.skip = skip
        self.concat = concat

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_size),
        )
        # self.up = nn.Sequential(
        #         nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),   # size up to specific
        #         nn.Conv2d(in_size, out_size, kernel_size=1),    # size maintain
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.BatchNorm2d(out_size),
        #     )

        if concat == True:
            self.conv_block = convBlock(in_size, out_size, padding, batch_norm)
        elif concat == False:
            self.conv_block = convBlock(out_size, out_size, padding, batch_norm)
        
    def forward(self, x, bridge):
        up = self.up(x)

        if self.skip == True:
            if self.concat == True:
                # print(up.shape)
                # print(bridge.shape)
                up = torch.cat([up, bridge], 1)
                out = self.conv_block(up)
            else:
                out = self.conv_block(up + bridge)
        
        else: 
            out = self.conv_block(up)

        return out

class Classifier(nn.Module):
    def __init__(self, in_channels=3, depth=4, wf=6, class_num=2, padding=True, batch_norm=True, skip=True, concat=True):
        super(Classifier, self).__init__()
        prev_channels = in_channels
        self.skip = skip
        self.concat = concat
        self.depth = depth
        self.down_path = nn.ModuleList()

        self.first = nn.Sequential(
            convBlock(prev_channels, 2 ** 5, padding, batch_norm)
        )
        prev_channels = 2 ** 5

        for i in range(depth):
            self.down_path.append(
                convBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, class_num, kernel_size=1),
            # nn.Sigmoid(), # cross entropy, BCEWithLogitsLoss 일때는 필요없음
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x, blocks):
        x = self.first(x)

        for i, down in enumerate(self.down_path):
            if self.skip == True:
                if self.concat == True:
                    x = down()
                elif self.concat == False:
                    x = down(x + blocks[i])     # skip with encoder
                    # x = down(x + blocks[self.depth - i - 1])     # skip with decoder
            else:
                x = down(x)

            if i != len(self.down_path) - 1:
                x = F.max_pool2d(x, 2)
        
        x = self.last(x)

        return x     

class sharedEncoder(nn.Module):
    def __init__(self, in_channels=3, depth=5, wf=5, class_num=2, padding=True, batch_norm=True):
        super(sharedEncoder, self).__init__()
        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(
                convBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, class_num, kernel_size=1),
            # nn.Sigmoid(), # cross entropy, BCEWithLogitsLoss 일때는 필요없음
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.max_pool2d(x, 2)
        bottleNeck = x

        x = self.last(x)

        return blocks, bottleNeck, x 