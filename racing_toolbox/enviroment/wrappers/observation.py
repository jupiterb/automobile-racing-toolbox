import gym
import numpy as np

from observation.config import LidarConfig, TrackSegmentationConfig

from observation.lidar import Lidar
from observation.track_segmentation import TrackSegmenter
import gym.spaces


class SqueezingWrapper(gym.ObservationWrapper):
    """This wrapper applies np.squeeze to make shape of observation compatible with stb3"""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.move_axis = lambda obs: np.moveaxis(obs, 0, -1)
        self.observation_space = gym.spaces.Box(
            0, 1, self.move_axis(env.observation_space.sample()).shape
        )

    def observation(self, observation: np.ndarray):
        observation = np.squeeze(observation)
        return self.move_axis(observation)


class RescaleWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, env.observation_space.shape)

    def observation(self, observation: np.ndarray):
        return observation / 255.0


class LidarWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: LidarConfig) -> None:
        super().__init__(env)
        self._lidar = Lidar(config)
        self.observation_space = gym.spaces.Box(
            0, 1, (len(range(*config.angles_range)) + 1, config.depth)
        )

    def observation(self, observation: np.ndarray):
        return self._lidar.scan_2d(observation)[0]


class TrackSegmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: TrackSegmentationConfig) -> None:
        super().__init__(env)
        self._track_segmenter = TrackSegmenter(config)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._track_segmenter.perform_segmentation(observation)
