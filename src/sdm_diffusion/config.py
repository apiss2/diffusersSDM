from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TrainingConfig:
    """Configuration values for training the semantic diffusion model."""

    data_root: str
    mask_root: str
    image_size: int = 256
    batch_size: int = 4
    num_classes: int = 19
    epochs: int = 1
    lr: float = 1e-4
    diffusion_steps: int = 1000
    schedule: str = "linear"


def load_config(path: str | Path) -> TrainingConfig:
    """Load a YAML config file into a :class:`TrainingConfig` instance."""

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    return TrainingConfig(**data)
