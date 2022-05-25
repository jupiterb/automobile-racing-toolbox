from __future__ import annotations
from pydantic import BaseModel
from enum import Enum
from pathlib import Path


class TrainingParameters(BaseModel):
    algorithm: str = "Foo"  # it is ignored for now
    tensorboard_dir_path: Path = Path(__file__) / "demo/tensorboard"
    log_dir_path: Path = Path(__file__) / "demo/trening"


class RLAlgorithm(Enum):
    DQN = "DQN"
    SAC = "SAC"


class AlgorithmParameters(BaseModel):
    timesteps: int = 2_000
