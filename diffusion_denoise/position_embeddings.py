# ### Position embeddings
#
# As the parameters of the neural network are shared across time (noise level),
# the authors employ sinusoidal position embeddings to encode $t$,
# inspired by the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).
# This makes the neural network "know" at which particular time step (noise level) it is operating,
# for every image in a batch.
#
# The `SinusoidalPositionEmbeddings` module takes a tensor of shape `(batch_size, 1)` as input
# (i.e. the noise levels of several noisy images in a batch),
# and turns this into a tensor of shape `(batch_size, dim)`,
# with `dim` being the dimensionality of the position embeddings.
# This is then added to each residual block, as we will see further.


import math

import torch
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings