import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import torch
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils import *
from stegan_block.Network import *

from utils.load_train_setting import *
import numpy as np

from stegan_block.blocks import ConvBNRelu


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

        self.weight = nn.Parameter(torch.ones(normalized_shape))
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



class FeedForward1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward1, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FeedForward2(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward2, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor) * 2

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class FeedForward3(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward3, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor) * 2
        # branch1: conv1*1(32)
        self.b1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        # branch2: conv1*1(32) --> con3*3(32)
        self.b2_1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.b2_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=bias)

        # branch3: conv1*1(32) --> conv3*3(32) --> conv3*3(32)
        self.b3_1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.b3_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=bias)
        self.b3_3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=bias)

        # totalbranch: conv1*1(256)
        self.tb = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b_out1 = self.b1(x)
        b_out2 = self.b2_2(self.b2_1(x))
        b_out3 = self.b3_3(self.b3_2(self.b3_1(x)))
        b_out = torch.cat([b_out1, b_out2, b_out3], dim=1)
        b_out = F.gelu(b_out)
        out = self.tb(b_out)
        return out


class FeedForward4(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward4, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * F.gelu(x2)
        x = self.project_out(x)
        return x

##########################################################################
## Channel Attention Block
class selfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(selfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class crossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(crossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, message):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        qkv_m = self.qkv_dwconv(self.qkv(message))
        q_m, k_m, v_m = qkv_m.chunk(3, dim=1)

        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_m = rearrange(q_m, 'bm (head cm) hm wm -> bm head cm (hm wm)', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q_m = torch.nn.functional.normalize(q_m, dim=-1)

        attn = (q_m @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = selfAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward1(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock2, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = selfAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward2(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock3(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock3, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = selfAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward3(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock4(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock4, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = selfAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward4(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

############################################
##cross Attention
class CT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CT, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = crossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward4(dim, ffn_expansion_factor, bias)

    def forward(self, x, message):
        x = x + self.attn1(self.norm1(x), self.norm1(message))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Secret_pre(nn.Module):
    def __init__(self, H, W, channels=64,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'
                 ):
        super(Secret_pre, self).__init__()
        self.H = H
        self.W = W

        self.secret_pre_layer = nn.Sequential(
            ConvBNRelu(3, 8),
            ConvBNRelu(8, 16),
            ConvBNRelu(16, 32),
            TransformerBlock4(dim=32, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            TransformerBlock4(dim=32, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            TransformerBlock4(dim=32, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            TransformerBlock4(dim=32, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
            TransformerBlock4(dim=32, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        )
        self.secret_first_layer = TransformerBlock4(dim=32, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, secret):
        secret_pre = self.secret_pre_layer(secret)
        intermediate2 = self.secret_first_layer(secret_pre)
        return intermediate2

##########################################################################
##---------- Restormer -----------------------
class Encoder(nn.Module):
    def __init__(self,
                 H = 128,
                 W = 128,
                 inp_channels=3,
                 out_channels=3,
                 dim=128,
                 num_blocks=[2, 2, 2, 2],
                 heads=[4, 4, 4, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'
                 ):

        super(Encoder, self).__init__()

        self.secret_pre = Secret_pre(H, W)
        self.CT1 = CT(dim=32, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                      LayerNorm_type=LayerNorm_type)
        self.CT2 = CT(dim=32, num_heads=heads[0],ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                      LayerNorm_type=LayerNorm_type)
        self.CT3 = CT(dim=32, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                      LayerNorm_type=LayerNorm_type)
        self.CT4 = CT(dim=32, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                      LayerNorm_type=LayerNorm_type)

        self.transformer_block1 = nn.Sequential(*[
            TransformerBlock4(dim=64, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.transformer_block2 = nn.Sequential(*[
            TransformerBlock4(dim=64, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.transformer_block3 = nn.Sequential(*[
            TransformerBlock4(dim=64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.transformer_block4 = nn.Sequential(*[
            TransformerBlock4(dim=64, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.keep_layers = nn.Sequential(
            ConvBNRelu(64, 64),
            ConvBNRelu(64, 64)
        )
        self.catconv1 = ConvBNRelu(64, 32)
        self.catconv2 = ConvBNRelu(32, 16)
        self.catconv3 = ConvBNRelu(16, 8)
        self.output = nn.Conv2d(8 + 3, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        if train_continue == False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.2)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0.001)

    def forward(self, cover, secret):
        secret_pre = self.secret_pre(secret)
        cover_pre = self.secret_pre(cover)
        ct1 = self.CT1(cover_pre, secret_pre)
        ct2 = self.CT2(ct1, secret_pre)
        concat1 = torch.cat([ct2, secret_pre], dim=1)
        concat1 = self.keep_layers(concat1)
        out_1 = self.transformer_block1(concat1)
        out_2 = self.transformer_block2(out_1)
        out_3 = self.transformer_block3(out_2)
        out_4 = self.transformer_block4(out_3)
        conv1 = self.catconv1(out_4)
        conv2 = self.catconv2(conv1)
        conv3 = self.catconv3(conv2)
        cat = torch.cat([conv3, cover], dim=1)
        stegan_image = self.output(cat)
        return stegan_image

