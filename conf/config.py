from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    #--GENERAL--
    #Output path
    output_path: Path = Path('./results/')
    #Model URL on HF
    model_path: str = 'CompVis/stable-diffusion-v1-4'
    #UNet config, a local modified version from diffusers
    unet_config: Path = Path('./conf/unet/config.json')
    #Image editing
    real_image_editing : bool = False

    #--INFERENCE--
    # Guiding text prompt
    prompt: str = ''
    # Bounding boxes
    bbox: List[list] = field(default_factory=lambda: [[], []])
    # Phrases for the bboxes
    phrases: List[str] = field(default_factory=lambda: ['', ''])
    # Seeds for generating noise
    seeds: List[int] = field(default_factory=lambda: [42])
    #Loss scale
    loss_scale: int = 30
    #Batch size
    batch_size: int = 1
    #Loss threshold
    loss_threshold: float = 0.2
    #Max iter
    max_iter: int = 5
    #Max index step
    max_index_step: int = 10
    #Max timesteps
    timesteps: int = 50
    #Guidance scale
    guidance_scale: float = 7.5
    #Random seed
    #rand_seed: int = 445

    #--NOISE SCHEDULE--
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    num_train_timesteps: int = 1000

    # def __post_init__(self):
    #     self.output_path.mkdir(exist_ok=True, parents=True)
