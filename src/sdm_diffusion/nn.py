import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


def checkpoint(func, inputs, parameters, use_checkpoint):
    if use_checkpoint:
        return cp.checkpoint(func, *inputs)
    else:
        return func(*inputs)


def conv_nd(dims, in_channels, out_channels, kernel_size, stride=1, padding=0):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    elif dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        raise ValueError(f"unsupported dims: {dims}")


def linear(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)


def avg_pool_nd(dims, kernel_size, stride=None):
    if dims == 1:
        return nn.AvgPool1d(kernel_size, stride=stride)
    elif dims == 2:
        return nn.AvgPool2d(kernel_size, stride=stride)
    elif dims == 3:
        return nn.AvgPool3d(kernel_size, stride=stride)
    else:
        raise ValueError(f"unsupported dims: {dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    return nn.GroupNorm(32, channels)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(0, half, dtype=th.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
