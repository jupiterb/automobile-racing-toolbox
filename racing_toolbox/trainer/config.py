from __future__ import annotations
from typing import Callable, Any
from pydantic import BaseModel, PositiveFloat, PositiveInt
from ray.rllib.offline.input_reader import InputReader
import gym 


class TrainingConfig(BaseModel):
    # TODO:
    # - Callbacks,
    # - test "remote_worker_envs" will it create env in client?, 
    # - how to configure trigger for video recorder?
    env: gym.Env 
    input: Callable[[Any], InputReader]
    num_workers: PositiveInt

    rollout_fragment_length: PositiveInt
    gamma: PositiveFloat=0.99 
    lr: PositiveFloat=1e-4 
    train_batch_size: PositiveFloat=200 

    log_level: str="INFO"
    framework: str="torch"
    model: ModelConfig

    # in rllib algorithm config is flatten on this level, but for readability made it nested
    algorithm: AlgorithmConfig


class ModelConfig(BaseModel):
    fcnet_hiddens: list[int] # number of units in hidden layers
    fcnet_activation: str 


class AlgorithmConfig(BaseModel):
    pass
    


class DQNConfig(AlgorithmConfig):
    v_min: float=-10
    v_max: float=10
    dueling: bool=True 
    double_q: bool=True  
    hiddens: list[int]=[256] # is this the same as model fcnet_hiddens?
    replay_buffer_config: ReplayBufferConfig     


class ReplayBufferConfig(BaseModel):
    capacity: PositiveInt=50_000
