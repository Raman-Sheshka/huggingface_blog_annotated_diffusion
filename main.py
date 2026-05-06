from diffusion_denoise.config import settings
from diffusion_denoise.logger import get_logger, setup_logging
from diffusion_denoise.model import DiffusionModel


logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    logger.info("Launching diffusion denoise training")

    model = DiffusionModel()
    model.train(settings.num_epochs)


if __name__ == "__main__":
    main()
