from typing import Callable, Any, Optional, Union
from ray.rllib.offline.input_reader import InputReader
from pydantic import validator
import gym
from gym.envs.registration import spec

from racing_toolbox.training.config.user_defined import TrainingConfig


class TrainingParams(TrainingConfig):
    class Config:
        arbitrary_types_allowed = True

    env: gym.Env
    input_: Optional[Callable[[Any], InputReader]] = None

    @validator("env", allow_reuse=True)
    def check_env(cls, v):
        if not isinstance(v, gym.Env):
            raise ValueError
        if not hasattr(v, "observation_space"):
            raise ValueError("given env doesnt have observation space defined")
        if not hasattr(v, "action_space"):
            raise ValueError("given env dosnt have action_space defined")
        return v
