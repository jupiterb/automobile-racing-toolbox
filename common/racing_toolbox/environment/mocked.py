import gym 
import gym.spaces 
from typing import Generic, TypeVar
from enum import Enum
import numpy as np 

_Action = TypeVar("_Action", bound=Enum, covariant=True)

class MockedEnv(gym.Env, Generic[_Action]):
    def __init__(self, action_mapping: dict[str, _Action], screen_shape: tuple[int, int]):
        self._available_actions = list(action_mapping)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._available_actions)],
            dtype=np.float16,
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(*screen_shape, 3),
            dtype=np.uint8,
        )
