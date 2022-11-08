from __future__ import annotations
from pydantic import BaseModel, PositiveFloat, PositiveInt

from racing_toolbox.trainer.config import ModelConfig


class UserDefinedBCConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    lr: PositiveFloat = 1e-4
    train_batch_size: PositiveFloat = 100
    num_iterations: PositiveInt = 3
    model: ModelConfig
