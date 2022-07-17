import gym
import numpy as np 
from collections import deque


class OffTrackPunishment(gym.RewardWrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, observation), done, info

    def reward(self, reward, observation):
        r = - abs(reward) if self._is_off_track(observation) else reward # abs to make sure it is still punishment
        return r

    def _is_off_track(self, observation) -> bool:
        """If car is off track, std of green channel is lower than 30"""
        return np.std(observation[:, :, 1]) < 35


class SpeedDropPunishment(gym.RewardWrapper):
    """This wrapper will add punishment for every 'major' drop in speed. So it assumes that returned reward is speed related"""

    def __init__(self, env, memory_length: int, diff_thresh: float) -> None:
        super().__init__(env)
        self.reward_history = deque([], maxlen=memory_length)
        self.threshold = diff_thresh

    def reward(self, reward: float) -> float:
        baseline = np.mean(self.reward_history)
        self.reward_history.append(reward)
        r = reward - (baseline - reward) ** 2 if reward < baseline - self.threshold else reward
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