from typing import Any
import gym
import numpy as np

from racing_toolbox.datatool.datasets import DatasetContainer


class DatasetBasedEnv(gym.Env):

    _is_final: bool
    _next_observation: np.ndarray
    _last_action: np.ndarray

    def __init__(self, datasets: DatasetContainer) -> None:
        super().__init__()
        self._datasets = datasets.get_all()
        self._next()

    @property
    def last_action(self):
        return self._last_action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self._last_action = action
        observation = self._next_observation
        try:
            self._next()
        except StopIteration:
            self._is_final = True
        return observation, 0.0, self._is_final, {}

    def _next(self):
        self._next_observation = self._datasets.__next__()[0]
        self._is_final = False
