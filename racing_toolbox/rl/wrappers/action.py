import gym
import numpy as np
from typing import Callable


class DiscreteActionToVectorWrapper(gym.ActionWrapper):
    def __init__(
        self, env: gym.Env, available_actions: list[set[str]], all_actions: list[str]
    ) -> None:
        super().__init__(env)
        self._available_actions = available_actions
        self.action_space = gym.spaces.Discrete(len(self._available_actions))
        self._all_actions = all_actions

    def action(self, action: int) -> np.ndarray:
        actions_set = self._available_actions[action]
        return np.array(
            [1.0 if action in actions_set else 0.0 for action in self._all_actions]
        )


class TransformActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        indexes_to_transform: list[int],
        transform_function: Callable[[float], float],
    ) -> None:
        super().__init__(env)
        self._to_transform = indexes_to_transform
        self._transform = transform_function

    def action(self, action: np.ndarray) -> np.ndarray:
        for index in self._to_transform:
            action[index] = self._transform(action[index])
        return action


class StandardActionRangeToPositiveWarapper(TransformActionWrapper):
    def __init__(
        self, env: gym.Env, indexes_to_be_positive: list[int], max_value: float = 1.0
    ) -> None:
        super().__init__(env, indexes_to_be_positive, lambda x: (x + max_value) / 2)


class ZeroThresholdingActionWrapper(TransformActionWrapper):
    def __init__(self, env: gym.Env, indexes_to_threshold: list[int]) -> None:
        super().__init__(env, indexes_to_threshold, lambda x: 1.0 if x > 0 else 0.0)


class SignSplitActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, to_split: dict[int, bool]) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(to_split)],
            dtype=np.float16,
        )
        self._to_split = to_split
        self._new_action_shape = len(to_split) + len(
            [split for split in to_split.values() if split]
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros(shape=(self._new_action_shape))
        new_action_index = 0
        for index, value in enumerate(action):
            if self._to_split[index]:
                new_action[new_action_index] = value if value > 0 else 0.0
                new_action_index += 1
                new_action[new_action_index] = -value if value < 0 else 0.0
            else:
                new_action[new_action_index] = value
            new_action_index += 1
        return new_action
