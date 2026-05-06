import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
import requests
import sys
from pathlib import Path

import torch

from diffusion_denoise.forward_diffusion import ForwardDiffusion
from diffusion_denoise.config import METRICS_DIR, settings
from diffusion_denoise.utils import get_image_transform, get_reverse_transform

# use seed for reproducability
torch.manual_seed(0)

transform = get_image_transform()
reverse_transform = get_reverse_transform()

def get_noisy_image(x_start, t):
  
  # add noise
  forward_diffusion_obj = ForwardDiffusion(timesteps=settings.timesteps,
                                     beta_start=settings.beta_start,
                                     beta_end=settings.beta_end)
  
  x_noisy = forward_diffusion_obj.q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

def show_sample(samples, step: int=-1, random_index:int = 5, show_plot: bool = True):
    
    if isinstance(samples, list):
        sample = samples[step][random_index]
    if isinstance(samples, torch.Tensor):
        sample = samples.cpu().detach()[random_index]    
    fig, ax = plt.subplots(figsize=(4, 4))
    fig = plt.imshow(sample.reshape(settings.image_size, settings.image_size, settings.channels), cmap="gray")
    ax.axis("off")
    if show_plot:
        plt.show()
    return fig


def plot_training_curve(metrics_path=None, save_plot=False, show_plot=True, dpi=150):
    if metrics_path is None:
        metrics_files = sorted(METRICS_DIR.glob("training_metrics_*.csv"))
        if not metrics_files:
            raise FileNotFoundError(f"No training metrics files found in {METRICS_DIR}")
        metrics_path = metrics_files[-1]
    else:
        metrics_path = Path(metrics_path)

    metrics = np.genfromtxt(metrics_path, delimiter=",", names=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["global_step"], metrics["loss"], color = "k")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    
    fig.tight_layout()

    if save_plot:
        output_path = metrics_path.with_name(f"{metrics_path.stem}_loss.png")
        fig.savefig(output_path, dpi=dpi)
    if show_plot:
        plt.show()
    return fig
  
    
# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot_example(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [imgs[0]] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
    
    return fig
  
def get_web_image(url: str = None):
    if url is None:
      url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    try:  
      image = Image.open(requests.get(url, stream=True).raw)
      return image
    except Exception as e:
      print(f"Error loading image from URL: {e}")
      return None  
       

def animated_sample(samples, random_index:int = 5, save_gif: bool = True, show_plot: bool = True):
    fig, ax = plt.subplots(figsize=(4,4))
    ims = []
    for i in range(settings.timesteps):
        im = plt.imshow(samples[i][random_index].reshape(settings.image_size,
                                                         settings.image_size,
                                                         settings.channels
                                                         ),
                        cmap="gray",
                        animated=True
        )
        ims.append([im])
    ax.axis("off")    
    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    if save_gif:
        animate.save('diffusion.gif')
    if show_plot:
        plt.show()
    
    return animate    

if __name__ == "__main__":
    
    image = get_web_image()
    x_start = transform(image).unsqueeze(0)
    print(x_start.shape)
    
    plot_example([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])
