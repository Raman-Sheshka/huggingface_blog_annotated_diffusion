import numpy as np
from inspect import isfunction

import torch
from torch import nn

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize,RandomHorizontalFlip

# use seed for reproducability

def get_image_transform(image_size: int = 128):
    return Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])


def get_reverse_transform():
    return Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.clamp(0, 1)),
        Lambda(lambda t: t.detach().cpu()),
        Lambda(lambda t: t.permute(1, 2, 0)),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])


def get_data_transform():
    return Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
    ])


def exists(x):
    """Check if a value exists (is not None).
    """
    return x is not None

def default(val, d):
    """Return the value if it exists,
    otherwise return a default value or the result of a default function.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)