import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from diffusion_denoise.config import PROJECT_DIR


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = PROJECT_DIR / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "diffusion_denoise.log"
CONSOLE_HANDLER_NAME = "diffusion_denoise_console"
FILE_HANDLER_NAME = "diffusion_denoise_file"


def _has_handler(logger: logging.Logger, handler_name: str) -> bool:
    return any(handler.get_name() == handler_name for handler in logger.handlers)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = DEFAULT_LOG_FILE,
) -> None:
    """Configure application logging from the entry point."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not _has_handler(root_logger, CONSOLE_HANDLER_NAME):
        console_handler = logging.StreamHandler()
        console_handler.set_name(CONSOLE_HANDLER_NAME)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file is None:
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.set_name(FILE_HANDLER_NAME)
    file_handler.setFormatter(formatter)
    if not _has_handler(root_logger, FILE_HANDLER_NAME):
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
