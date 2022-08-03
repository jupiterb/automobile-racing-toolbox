from charset_normalizer import detect
import gym
import numpy as np
from collections import deque
from typing import Callable

from interface import GameInterface
from rl.config import EventDetectionParameters
from rl.event import EventDetector


class OffTrackPunishment(gym.RewardWrapper):
    def __init__(self, env, metric: Callable[[float], float]):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, observation), done, info

    def reward(self, reward, observation):
        r = (
            self.metric(reward) if self._is_off_track(observation) else reward
        )  # abs to make sure it is still punishment
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
        metric: Callable[[float], float],
    ) -> None:
        super().__init__(env)
        self.reward_history = deque([], maxlen=memory_length)
        self.threshold = diff_thresh
        self.metric = metric

    def reward(self, reward: float) -> float:
        baseline = np.mean(self.reward_history)
        self.reward_history.append(reward)
        r = (
            reward - self.metric(baseline - reward)
            if reward < baseline - self.threshold
            else reward
        )
        return r


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_value: float, max_value: float) -> None:
        super().__init__(env)
        self.min_value = min_value
        self.max_value = max_value

    def reward(self, reward: float) -> float:
        if reward < self.min_value:
            r = self.min_value
        elif reward > self.max_value:
            r = self.max_value
        else:
            r = reward
        return r


class StandarizeReward(gym.RewardWrapper):
    def __init__(self, env, baseline: float, scale: float) -> None:
        super().__init__(env)
        self.baseline = baseline
        self.scale = scale

    def reward(self, reward: float) -> float:
        r = (reward - self.baseline) / self.scale
        return r


class CheckpointReward(gym.RewardWrapper):
    def __init__(
        self,
        env: gym.Env,
        detector_parameters: list[EventDetectionParameters],
        game_interface: GameInterface,
    ) -> None:
        super().__init__(env)
        self._detector = EventDetector(detector_parameters)
        self._game_interface = game_interface

    def reward(self, reward: float) -> float:
        if self._detector.is_final(self._game_interface.perform_ocr()):
            self._detector.reset()
            print("CHECKPOINT!")
            # TODO: reward based on checkpoint
        return reward
