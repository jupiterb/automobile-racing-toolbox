from __future__ import annotations
from collections import namedtuple
from typing import Callable, Any, Literal, Optional, Union
from pydantic import BaseModel, PositiveFloat, PositiveInt, validator, Field
from ray.rllib.offline.input_reader import InputReader
import gym


Activation = Literal["relu", "tanh", "sigmoid"]
ConvLayerTH = namedtuple("ConvLayerTH", ["out_channels", "kernel", "stride"])


class ReplayBufferConfig(BaseModel):
    capacity: PositiveInt = 50_000


class ModelConfig(BaseModel):
    fcnet_hiddens: list[int]  # number of units in hidden layers
    fcnet_activation: str
    conv_filters: list[ConvLayerTH] = []
    conv_activation: list[Activation] = []


class AlgorithmConfig(BaseModel):
    pass


class DQNConfig(AlgorithmConfig):
    v_min: float = -10
    v_max: float = 10
    dueling: bool = True
    double_q: bool = True
    hiddens: list[int] = [256]
    replay_buffer_config: ReplayBufferConfig


class TrainingConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # TODO:
    # - Callbacks,
    # - test "remote_worker_envs" will it create env in client?,
    # - how to configure trigger for video recorder?
    env: Union[gym.Env, str]
    input_: Optional[Callable[[Any], InputReader]]
    num_rollout_workers: int = Field(ge=0)
    rollout_fragment_length: PositiveInt
    compress_observations: bool = False
    gamma: PositiveFloat = 0.99
    lr: PositiveFloat = 1e-4
    train_batch_size: PositiveFloat = 200
    max_iterations: PositiveInt = float("inf")

    log_level: str = "INFO"
    framework: str = "torch"
    model: ModelConfig

    # in rllib algorithm config is flatten on this level, but for readability made it nested
    algorithm: DQNConfig  # TODO: when more algorithm will be available, make it union

    @validator("env")
    def check_env(cls, v):
        if isinstance(v, str):
            return v

        if not isinstance(v, gym.Env):
            raise ValueError
        if not hasattr(v, "observation_space"):
            raise ValueError("given env doesnt have observation space defined")
        if not hasattr(v, "action_space"):
            raise ValueError("given env dosnt have action_space defined")
        return v
