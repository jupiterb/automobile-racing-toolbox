from __future__ import annotations
from pydantic import BaseModel
from enum import Enum
from pathlib import Path


class TrainingParameters(BaseModel):
    algorithm: str = "DQN"  # it is ignored for now
    neural_network_architecture: list[int]=[32, 32, 10]
    tensorboard_dir_path: Path = Path(__file__).parent / "demo/tensorboard"
    log_dir_path: Path = Path(__file__).parent / "demo/trening"


class RLAlgorithm(Enum):
    DQN = "DQN"
    SAC = "SAC"


class AlgorithmParameters(BaseModel):
    timesteps: int = 2_000
