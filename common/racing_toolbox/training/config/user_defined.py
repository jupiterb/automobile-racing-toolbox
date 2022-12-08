from __future__ import annotations
from collections import namedtuple
from typing import Literal, Union, Optional
from pydantic import BaseModel, PositiveFloat, PositiveInt, Field
from pathlib import Path


Activation = Literal["relu", "tanh", "sigmoid"]
ConvLayerTH = namedtuple("ConvLayerTH", ["out_channels", "kernel", "stride"])


class ReplayBufferConfig(BaseModel):
    capacity: PositiveInt = 50_000


class ModelConfig(BaseModel):
    fcnet_hiddens: list[int]  # number of units in hidden layers
    fcnet_activation: Activation
    conv_filters: list[
        tuple[PositiveInt, tuple[PositiveInt, PositiveInt], PositiveInt]
    ] = []
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


class SACConfig(AlgorithmConfig):
    twin_q: bool = True
    tau: float = 5e-3
    initial_alpha: float = 1.0
    q_model_config: Optional[ModelConfig] = None
    policy_model_config: Optional[ModelConfig] = None
    replay_buffer_config: ReplayBufferConfig

    class Config:
        frozen = True


class BCConfig(AlgorithmConfig):
    pass


class EvalConfig(BaseModel):
    eval_name: Optional[str]
    eval_interval_frequency: PositiveInt
    eval_duration: PositiveInt
    eval_duration_unit: Literal["episodes", "timesteps"] = "timesteps"


class TrainingConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    num_rollout_workers: int = Field(ge=0)
    rollout_fragment_length: PositiveInt
    compress_observations: bool = False
    gamma: PositiveFloat = 0.99
    lr: PositiveFloat = 1e-4
    train_batch_size: PositiveInt = 200
    max_iterations: PositiveInt = 100
    stop_reward: float = 1e6
    checkpoint_frequency: PositiveInt = 10

    evaluation_config: Optional[EvalConfig] = None

    log_level: str = "INFO"
    model: ModelConfig

    # in rllib algorithm config is flatten on this level, but for readability made it nested
    algorithm: Union[DQNConfig, SACConfig, BCConfig]
    offline_data: Optional[list[Path]] = None
