from __future__ import annotations
import gym
import math
import numpy as np
from collections import deque
from typing import Callable
from racing_toolbox.environment.utils.logging import log_reward


class OffTrackPunishment(gym.RewardWrapper):
    def __init__(self, env, off_track_reward: float, terminate: bool):
        super().__init__(env)
        self.off_track_reward = off_track_reward
        self.terminate = terminate

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.terminate and self._is_off_track(observation):
            return observation, 0, True, info
        return observation, self.reward(reward, observation), done, info

    @log_reward(__name__)
    def reward(self, reward, observation):
        r = self.off_track_reward if self._is_off_track(observation) else reward
        return r

    def _is_off_track(self, observation) -> bool:
        """If car is off track, std of green channel is lower than 30"""
        # when thsis will be properly implemented, it will be configurable also
        return np.std(observation[:, :, 1]) < 35


class SpeedDropPunishment(gym.RewardWrapper):
    """This wrapper will add punishment for every 'major' drop in speed. So it assumes that returned reward is speed related"""

    def __init__(
        self,
        env,
        memory_length: int,
        diff_thresh: float,
        exponent: float,
        only_diff: bool = False,
    ) -> None:
        super().__init__(env)
        self.reward_history = deque([0], maxlen=memory_length)
        self.threshold = diff_thresh
        self.metric = lambda x: x**exponent

        if only_diff:
            self.get_base_reward = lambda r: 0
        else:
            self.get_base_reward = lambda r: r

    @log_reward(__name__)
    def reward(self, reward: float) -> float:
        baseline = np.mean(self.reward_history)
        self.reward_history.append(reward)
        diff = reward - baseline
        base = self.get_base_reward(
            reward
        )  # skip speed itself and praise/punish only deviations
        if abs(diff) < self.threshold or diff == 0:
            return base
        r = base + math.copysign(self.metric(abs(diff)), diff)
        return r


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_value: float, max_value: float) -> None:
        super().__init__(env)
        self.min_value = min_value
        self.max_value = max_value

    @log_reward(__name__)
    def reward(self, reward: float) -> float:
        r = max(min(reward, self.max_value), self.min_value)
        return r


class StandarizeReward(gym.RewardWrapper):
    def __init__(self, env, baseline: float, scale: float) -> None:
        super().__init__(env)
        self.baseline = baseline
        self.scale = scale

    @log_reward(__name__)
    def reward(self, reward: float) -> float:
        r = (reward - self.baseline) / self.scale
        return r
