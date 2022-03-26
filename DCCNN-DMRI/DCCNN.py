import torch
import torch.nn as nn

import numpy as np

import modules

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

# conv blocks for CNNs
def conv_block(n_ch,                # input channel
                n_dims,                 # layers of CNNs
                n_feats=32,            # feature layers
                kernel_size=3,         # CNN kernel size
                dilation=1,            # dilated convolution
                bn=False,               # batchnormalization
                nl='lrelu',            # activation function
                conv_dim=2,             # 2DConv or 3DConv
                n_out=None):           # output channel

    # decide to use 2D or 3D
    if conv_dim == 2:
        conv = nn.Conv2d
    elif conv_dim == 3:
        conv = nn.Conv3d
    else:
        print("Wrong conv dimension.")

    # define output channel
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    # define activation function
    af = relu if nl == "relu" else lrelu

    # define CNNs in the middle
    def conv_i():
        return conv(n_feats, n_feats, kernel_size, stride=1,
                    padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, n_feats, kernel_size, stride=1,
                  padding=pad_conv, bias=True)
    conv_n = conv(n_feats, n_out, kernel_size, stride=1,
                  padding=pad_conv, bias=True)

    # fill in the CNNs
    layers = [conv_1, af()]

    for i in range(n_dims - 2):
        layers += [conv_i(), af(), nn.BatchNorm2d(n_feats) if bn]

    layers += [conv_n]

    return nn.Sequential(*layers)


class DCCNN(nn.Module):
    def __init__(self,
                 nc_dims=5,             # the depth of cascade
                 nd_dims=5,             # the number of convolution layers
                 n_channels=2):
        super(DCCNN, self).__init__()

        self.nc = nc_dims
        self.nd = nd_dims

        print('A cascade network with {} CNNs and {} DCs'.format(nc_dims, nd_dims))

        blocks = []
        dcs = []

        for i in range(nc_dims):
            # add the numbers of convolution layers
            blocks.append(conv_block(n_channels, nd_dims))
            dcs.append(modules.DataConsistencyInKspace(norm='ortho'))

        self.blocks = nn.ModuleList(blocks)
        self.dcs = dcs

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.blocks[i](x)
            x += x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x



