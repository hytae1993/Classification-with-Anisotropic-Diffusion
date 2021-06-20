import torch
from torch import nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, depth=5, wf=5, padding=False, batch_norm=True, up_mode='upsample',):
        super(Encoder, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)   # last channel = 2**(5+4) = 512
    
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
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        
        return blocks, x

class Decoder(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, depth=5, wf=5, padding=False, batch_norm=True, up_mode='upsample',last=1, skip=True, concat=True):
        super(Decoder, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.lastChannel = last
        self.size = [220,106,49,20]
        prev_channels = 512

        self.up_path = nn.ModuleList()

        for i in reversed(range(depth-1)):
            self.up_path.append(
                UNetUpBlock(self.size[i], prev_channels, 2 ** (wf+i), up_mode, padding, batch_norm, skip=skip, concat=concat)
            )
            prev_channels = 2 ** (wf+i)

        self.last = nn.Sequential(
            nn.Upsample(mode='bilinear', size=(224,224), align_corners=False),
            nn.Conv2d(prev_channels, self.lastChannel, kernel_size=1),
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
        x = bottleNeck 
        for i, up in enumerate(self.up_path):
            if blocks != None:
                x = up(x, blocks[-i - 1])
            elif blocks == None:
                x = up(x)
        mask = self.last(x)
        # print(mask.max(), mask.min())

        return mask

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU(inplace=True))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU(inplace=True))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, fe_size, in_size, out_size, up_mode, padding, batch_norm, skip=True, concat=True):
        super(UNetUpBlock, self).__init__()
        self.skip = skip
        self.concat = concat
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)    # size twice
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', size=(fe_size, fe_size), align_corners=False),   # size up to specific
                nn.Conv2d(in_size, out_size, kernel_size=1),    # size maintain
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_size),
            )

        if concat:
            self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm) # size i-2
        else:
            self.conv_block = UNetConvBlock(out_size, out_size, padding, batch_norm) # size i-2

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)

        # no connection
        if self.skip == False:
            out = self.conv_block(up)
        
        else:
            # concat connection
            if self.concat:
                up = torch.cat([up, bridge], 1)
                out = self.conv_block(up)

            # add connection
            else:
                out = self.conv_block(up + bridge)

        return out