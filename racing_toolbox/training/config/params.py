from typing import Callable, Any, Optional, Union
from ray.rllib.offline.input_reader import InputReader
from pydantic import validator
from gym import spaces
from gym.envs.registration import spec

from racing_toolbox.training.config.user_defined import TrainingConfig


class TrainingParams(TrainingConfig):
    class Config:
        arbitrary_types_allowed = True

    observation_space: spaces.Space
    action_space: spaces.Space
    env_name: Optional[str] = None
    input_: Optional[Union[Callable[[Any], Optional[InputReader]], list[str]]] = None

    # @validator("env", allow_reuse=True)
    # def check_env(cls, v):
    #     if isinstance(v, str):
    #         return v
    #     return v

    @validator("observation_space", "action_space", allow_reuse=True)
    def check_space(cls, val):
        assert isinstance(val, spaces.Space)
        return val
