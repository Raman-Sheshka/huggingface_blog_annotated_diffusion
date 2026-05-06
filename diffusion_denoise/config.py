from pathlib import Path

from pydantic import BaseModel
import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "settings.yml"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok = True)

CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok = True)

METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok = True)

SAMPLES_DIR = RESULTS_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok = True)

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok = True)

class Config(BaseModel):
    timesteps: int
    image_size: int
    channels: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    beta_start: float
    beta_end: float
    save_and_sample_every : int
    
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with Path(config_path).open("r") as f:
        config_dict = yaml.safe_load(f)    
    return Config(**config_dict)


settings = load_config()


if __name__ == "__main__":
    print(f"Project root directory: {PROJECT_DIR}")  
    print(settings)      
