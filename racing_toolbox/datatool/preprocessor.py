from re import I
import gym
import numpy as np
from typing import Optional, Callable

from racing_toolbox.datatool import DatasetContainer
from racing_toolbox.datatool.services import AbstractDatasetService
from racing_toolbox.datatool.utils import DatasetBasedEnv


def preprocess(
    source: DatasetContainer,
    destination: AbstractDatasetService,
    revert_action: Callable[[np.ndarray], dict[str, float]],
    wrapp_observations: Optional[Callable[[gym.Env], DatasetBasedEnv]] = None,
):
    env = DatasetBasedEnv(source)
    env = wrapp_observations(env) if wrapp_observations else env
    env.reset()

    def step():
        try:
            observation, _, final, _ = env.step(None)
            action = revert_action(env.last_action)
            destination.put(observation, action)
            return final
        except AssertionError:
            # needed for make FrameStack wrapper working
            return step()

    final = step()
    while not final:
        final = step()
