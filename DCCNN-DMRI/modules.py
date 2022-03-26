import torch
import torch.nn as nn

import numpy as np

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    :param k: input in k-space
    :param k0: initially sampled elements in k-space
    :param mask: corresponding non-zero location
    :param noise_lvl:
    :return:
    """

    if noise_lvl:
        out = (1 - mask) * k + mask * (k + noise_lvl * k0) / (1 + noise_lvl)
    else:
        out = (1 - mask) * k + mask * k0

    return out

class DataConsistencyInKspace(nn.Module):
    """
    Create data consistency operator


    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()

        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        :param x: input in image domain, of shape (n, 2, nx, ny, nt)
        :param k0: initially sampled elements in k-space
        :param mask: corresponding nonzero location
        :return:
        """

        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)

        elif x.dim() == 5:
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = torch.fft.fft(x, 2, normalized=self.normalized)
        #k = torch.fft.fft2(x, )
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.fft.ifft(out, 2, normalized=self.normalized)
        #x_res = torch.fft.ifft2(out, 2, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res

