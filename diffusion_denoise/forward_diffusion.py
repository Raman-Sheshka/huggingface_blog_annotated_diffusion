# ===========the forward diffusion process===============================
#
# The forward diffusion process gradually adds noise to an image from the real distribution,
# in a number of time steps $T$. This happens according to a **variance schedule**.
# The original DDPM authors employed a linear schedule:
#
# > We set the forward process variances to constants increasing linearly
# from $\beta_1 = 10^{−4}$ to $\beta_T = 0.02$.
#
# However, it was shown in ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672))
# that better results can be achieved when employing a cosine schedule. 
#
# Below, we define various schedules for the $T$ timesteps, as well as corresponding variables
import torch
import torch.nn.functional as F

from diffusion_denoise.config import settings
from diffusion_denoise.utils import get_reverse_transform

timesteps = settings.timesteps
reverse_transform = get_reverse_transform()

def cosine_beta_schedule(timesteps, s=0.008, beta_start=0.0001, beta_end=0.9999):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

#Importantly, we also define an `extract` function,
# which will allow us to extract the appropriate $t$ index for a batch of indices.

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

#To start with, let's use the linear schedule for $T=200$ time steps
# and define the various variables from the $\beta_t$
# which we will need, such as the cumulative product of the variances $\bar{\alpha}_t$.
# Each of the variables below are just 1-dimensional tensors,
# storing values from $t$ to $T$.


class ForwardDiffusion:
    def __init__(self, timesteps=200, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_noisy_image(self, x_start, t):
  
        # add noise
        
        x_noisy = self.q_sample(x_start, t=t)

        # turn back into PIL image
        noisy_image = reverse_transform(x_noisy.squeeze())

        return noisy_image
