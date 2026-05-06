from datasets import load_dataset, load_from_disk

from torch.utils.data import DataLoader


from diffusion_denoise.utils import get_data_transform
from diffusion_denoise.config import settings, DATA_DIR
from diffusion_denoise.logger import get_logger


logger = get_logger(__name__)

# define image transformations (e.g. using torchvision)
transform = get_data_transform()


# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

def get_dataloader():
    
    logger.info("Loading dataset and creating dataloader")
    try:
        dataset = load_from_disk(DATA_DIR / "fashion_mnist_local")
        logger.info(f"Loaded dataset from disk at {DATA_DIR / 'fashion_mnist_local'}")
    except Exception as e:
        logger.warning(f"Could not load dataset from disk: {e}. Loading from hub instead.")
        dataset = load_dataset("fashion_mnist")
        dataset.save_to_disk(DATA_DIR / "fashion_mnist_local")
        logger.info(f"Saved dataset to disk at {DATA_DIR / 'fashion_mnist_local'}")
         
    transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

    # create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=settings.batch_size, shuffle=True)
    logger.info(f"Created dataloader with batch size {settings.batch_size}")
    return dataloader
