from __future__ import annotations
from collections import namedtuple
from typing import Literal
from pydantic import BaseModel, PositiveFloat, PositiveInt, Field


Activation = Literal["relu", "tanh", "sigmoid"]
ConvLayerTH = namedtuple("ConvLayerTH", ["out_channels", "kernel", "stride"])


class ReplayBufferConfig(BaseModel):
    capacity: PositiveInt = 50_000


class ModelConfig(BaseModel):
    fcnet_hiddens: list[int]  # number of units in hidden layers
    fcnet_activation: Activation
    conv_filters: list[tuple[PositiveInt, PositiveInt, PositiveInt]] = []
    conv_activation: Activation = "relu"


class AlgorithmConfig(BaseModel):
    pass


class DQNConfig(AlgorithmConfig):
    v_min: float = -10
    v_max: float = 10
    dueling: bool = True
    double_q: bool = True
    hiddens: list[int] = []
    replay_buffer_config: ReplayBufferConfig

    class Config:
        frozen = True


class TrainingConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # TODO:
    # - Callbacks,
    # - test "remote_worker_envs" will it create env in client?,
    # - how to configure trigger for video recorder?
    num_rollout_workers: int = Field(ge=0)
    rollout_fragment_length: PositiveInt
    compress_observations: bool = False
    gamma: PositiveFloat = 0.99
    lr: PositiveFloat = 1e-4
    train_batch_size: PositiveInt = 200
    max_iterations: PositiveInt = 100
    stop_reward: float = float("inf")

    log_level: str = "INFO"
    model: ModelConfig

    # in rllib algorithm config is flatten on this level, but for readability made it nested
    algorithm: DQNConfig  # TODO: when more algorithm will be available, make it union
