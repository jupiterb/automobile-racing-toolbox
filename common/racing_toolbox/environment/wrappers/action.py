import gym
import numpy as np


class DiscreteActionToVectorWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, available_actions: dict[str, set[int]]) -> None:
        super().__init__(env)
        actions_number = (
            max([max(usages) for usages in available_actions.values() if any(usages)])
            + 1
        )
        # for empty action:
        actions_number += 1

        self.action_space = gym.spaces.Discrete(actions_number)
        self._actions = np.array(
            [
                [1.0 if action in usages else 0.0 for action in range(actions_number)]
                for usages in available_actions.values()
            ]
        ).T

    def action(self, action: int) -> np.ndarray:
        try:
            return self._actions[action]
        except IndexError:
            raise ValueError(
                f"Action {action} not supported. Supported actions are from 0 to {len(self._actions) - 1}"
            )

    def reverse_action(self, action: np.ndarray) -> int:
        matches = [i for i, act in enumerate(self._actions) if (act == action).all()]
        if len(matches):
            return matches[0]
        raise ValueError(
            f"Action {action} not supported. Supported actions are {self._actions}"
        )


class SplitBySignActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, to_split_index: int) -> None:
        assert isinstance(env.action_space, gym.spaces.Box)
        super().__init__(env)
        self._original_action_space_shape = env.action_space.shape
        action_size = self._original_action_space_shape[0] - 1
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
