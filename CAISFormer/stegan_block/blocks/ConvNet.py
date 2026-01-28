import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import torch
import numpy as np


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 将不可训练的的类型转换为可训练的类型
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ConvBNRelu(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
		super(ConvBNRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
			# nn.BatchNorm2d(channels_out),
			LayerNorm(channels_out, LayerNorm_type='WithBias'),
			nn.LeakyReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ConvBNReluOut(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride=1):
		super(ConvBNReluOut, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.BatchNorm2d(channels_out),
			nn.Tanh()
		)

	def forward(self, x):
		return self.layers(x)


class ConvNet(nn.Module):
	'''
	Network that composed by layers of ConvBNRelu
	'''

	def __init__(self, in_channels, out_channels, blocks):
		super(ConvNet, self).__init__()

		layers = [ConvBNRelu(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvBNRelu(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
