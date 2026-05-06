import csv
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from torchvision.utils import save_image

from diffusion_denoise.unet import Unet
from diffusion_denoise.forward_diffusion import ForwardDiffusion, extract
from diffusion_denoise.config import CHECKPOINTS_DIR, METRICS_DIR, SAMPLES_DIR, settings
from diffusion_denoise.data_loader import get_dataloader
from diffusion_denoise.logger import get_logger


device = "cuda" if torch.cuda.is_available() else "cpu"

logger = get_logger(__name__)

logger.info(f"Using device: {device}")

logger.info(f"Initializing forward diffusion object ({settings.timesteps},{settings.beta_start},{settings.beta_end})")

forward_diffusion = ForwardDiffusion(timesteps=settings.timesteps,
                                         beta_start=settings.beta_start,
                                         beta_end=settings.beta_end)

logger.info("Initializing dataloader")
dataloader = get_dataloader()


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device
        self.model = None
        self.optimizer = None
        self.dataloader = None
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_path = METRICS_DIR / f"training_metrics_{self.run_tag}.csv"
        self.model_path = CHECKPOINTS_DIR / f"model_{self.run_tag}.pth"
        
        
    def init_model(self):
        self.model = Unet(
            dim=settings.image_size,
            channels=settings.channels,
            dim_mults=(1, 2, 4,)
        )
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=settings.learning_rate)
        
    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model not initialized. Please train or load the model before saving.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Saved model checkpoint to %s", path)
        return
    
        
    def train(self, epochs):
        
        self.init_model()
        self.dataloader = dataloader
        global_step = 0
        logger.info(
            "Starting training: epochs=%s, device=%s, image_size=%s, channels=%s, batch_size=%s",
            epochs,
            self.device,
            settings.image_size,
            settings.channels,
            settings.batch_size,
        )

        logger.info("Writing training metrics to %s", self.metrics_path)
        with self.metrics_path.open("w", newline="") as metrics_file:
            metrics_writer = csv.DictWriter(
                metrics_file,
                fieldnames=["epoch", "step", "global_step", "loss", "learning_rate"],
            )
            metrics_writer.writeheader()
            
            for epoch in range(epochs):
                logger.info("Epoch %s of %s", epoch + 1, epochs)
                for step, batch in enumerate(self.dataloader):
                    self.optimizer.zero_grad()

                    batch_size = batch["pixel_values"].shape[0]
                    batch = batch["pixel_values"].to(self.device)

                    # Algorithm 1 line 3: sample t uniformally for every example in the batch
                    t = torch.randint(0, settings.timesteps, (batch_size,), device=self.device).long()

                    loss = p_losses(self.model, batch, t, loss_type="huber")
                    loss_value = loss.item()

                    metrics_writer.writerow(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "global_step": global_step,
                            "loss": loss_value,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )

                    if step % 100 == 0:
                        metrics_file.flush()
                        logger.info(f"epoch: {epoch + 1} step: {step} loss: {loss_value:.6f}")

                    loss.backward()
                    self.optimizer.step()

                    # save generated images
                    if step != 0 and step % settings.save_and_sample_every == 0:
                        milestone = step // settings.save_and_sample_every
                        batches = num_to_groups(4, batch_size)
                        all_images_list = list(
                            map(lambda n: sample(self.model,
                                                 image_size=settings.image_size,
                                                 batch_size=n,
                                                 channels=settings.channels
                                                 ), batches
                                )
                        )
                        all_images = torch.cat(all_images_list, dim=0)
                        all_images = (all_images + 1) * 0.5
                        sample_path = SAMPLES_DIR / f'sample_{self.run_tag}_{milestone}.png'
                        save_image(all_images, str(sample_path), nrow = 6)
                        logger.info(f"Saved generated sample grid to {sample_path}")

                    global_step += 1
        
        self.save_model(self.model_path)             
        return
    
    def load_model(self, path):
        self.init_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Loaded model checkpoint from {path}")
        return

    def load_latest_model(self):
        checkpoints = sorted(CHECKPOINTS_DIR.glob("model_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No model checkpoints found in {CHECKPOINTS_DIR}")

        latest_checkpoint = checkpoints[-1]
        self.load_model(latest_checkpoint)
        return latest_checkpoint
    
    def inference(self, num_samples=64):
        if self.model is None:
            logger.error("Model not initialized. Please train or load the model before inference.")
            raise ValueError("Model not initialized. Please train the model first.")
        
        samples = sample(self.model,
                         image_size=settings.image_size,
                         batch_size=num_samples,
                         channels=settings.channels)     
            
        return samples
    
    
                       
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = forward_diffusion.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(forward_diffusion.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        forward_diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(forward_diffusion.sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(forward_diffusion.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in range(0, settings.timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr 
