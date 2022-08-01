import gym
import numpy as np

from observation.config import LidarConfig
from observation.lidar import Lidar


class SqueezingWrapper(gym.ObservationWrapper):
    """This wrapper applies np.squeeze to make shape of observation compatible with stb3"""

    def observation(self, observation: np.ndarray):
        observation = np.squeeze(observation)
        return observation


class RescaleWrapper(gym.ObservationWrapper):
    def observation(self, observation: np.ndarray):
        return observation / 255.0


class LidarWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, lidar_config: LidarConfig) -> None:
        self._lidar = Lidar(lidar_config)
        super().__init__(env)

    def observation(self, observation: np.ndarray):
        return self._lidar.scan_2d(observation)[0]
