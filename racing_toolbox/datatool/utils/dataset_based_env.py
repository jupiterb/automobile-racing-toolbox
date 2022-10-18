from typing import Optional, Any
import gym
import numpy as np

from racing_toolbox.datatool import DatasetContainer
from racing_toolbox.observation.utils.ocr import OcrTool


class DatasetBasedEnv(gym.Env):

    _is_final: bool
    _next_observation: np.ndarray
    _last_action: np.ndarray

    def __init__(
        self, container: DatasetContainer, ocr_tool: OcrTool, reward_key: str = "speed"
    ) -> None:
        super().__init__()
        self._container = container
        self._ocr = ocr_tool
        self._reward_key = reward_key
        self._datasets = container.get_all()
        self._next()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self._next_observation.shape,
            dtype=np.uint8,
        )

    @property
    def last_action(self):
        return self._last_action

    def reset(self):
        self._datasets = self._container.get_all()
        self._next()
        return self._next_observation

    def step(self, _: Optional[Any] = None) -> tuple[np.ndarray, float, bool, dict]:
        observation = self._next_observation
        reward = self._ocr.perform(observation)[self._reward_key]
        try:
            self._next()
        except StopIteration:
            self._is_final = True
        return observation, reward, self._is_final, {}

    def _next(self):
        self._next_observation, self._last_action = self._datasets.__next__()
        # may be needed for some wrappers
        self._next_observation = np.float32(self._next_observation)
        self._is_final = False
