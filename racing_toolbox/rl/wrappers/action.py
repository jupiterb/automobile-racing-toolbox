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


class SplitBySignActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, to_split_index: int) -> None:
        assert isinstance(env.action_space, gym.spaces.Box)
        super().__init__(env)
        self._original_action_space_shape = env.action_space.shape
        action_size = len(self._original_action_space_shape[0]) - 1
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[action_size],
            dtype=np.float16,
        )
        self._split = to_split_index

        def copy_action_to_split(action, new_action):
            new_action[self._split + 2 :] = action[self._split + 1 :]

        def copy_action_from_split(action, new_action):
            new_action[self._split + 2 :] = action[self._split + 1 :]

        def copy_action_to_and_from_split(action, new_action):
            copy_action_to_split(action, new_action)
            copy_action_from_split(action, new_action)

        self._copy_action = lambda action, new_action: None
        if action_size > 1:
            if self._split == 0:
                self._copy_action = copy_action_from_split
            elif self._split == action_size - 1:
                self._copy_action = copy_action_to_split
            else:
                self._copy_action = copy_action_to_and_from_split

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros(shape=(self._original_action_space_shape))
        self._copy_action(action, new_action)
        value = action[self._split]
        new_action[self._split] = value if value > 0 else 0.0
        new_action[self._split + 1] = -value if value < 0 else 0.0
        return new_action
