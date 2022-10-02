import gym
import numpy as np
from typing import Optional, Callable, Generator

from racing_toolbox.datatool import DatasetContainer
from racing_toolbox.datatool.utils import DatasetBasedEnv


def preprocess(
    source: DatasetContainer,
    revert_action: Callable[[np.ndarray], dict[str, float]],
    wrapp_observations: Optional[Callable[[gym.Env], DatasetBasedEnv]] = None,
) -> Generator[tuple[np.ndarray, dict[str, float]], None, None]:
    env = DatasetBasedEnv(source)
    env = wrapp_observations(env) if wrapp_observations else env
    env.reset()

    done = False
    while not done:
        try:
            observation, _, done, _ = env.step(None)
            action = revert_action(env.last_action)
            yield observation, action
        except AssertionError:
            # needed for make FrameStack wrapper working
            continue
