from typing import Any
import gym
import numpy as np

from racing_toolbox.datatool import RecordingsContainer


class RecordingsBasedEnviroment(gym.Env):

    _is_final: bool
    _next_observation: np.ndarray

    def __init__(self, recordings: RecordingsContainer) -> None:
        super().__init__()
        self._recordings = recordings.get_all()
        self._next()

    def step(self, action: Any | None = None) -> tuple[np.ndarray, float, bool, dict]:
        observation = self._next_observation
        try:
            self._next()
        except StopIteration:
            self._is_final = True
        return observation, 0.0, self._is_final, {}

    def _next(self):
        self._next_observation = self._recordings.__next__()[0]
        self._is_final = False
