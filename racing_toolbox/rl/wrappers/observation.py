import gym
import numpy as np

from racing_toolbox.observation.config import LidarConfig, TrackSegmentationConfig

from racing_toolbox.observation.lidar import Lidar
from racing_toolbox.observation.track_segmentation import TrackSegmenter
from racing_toolbox.rl.utils.logging import log_observation


class SqueezingWrapper(gym.ObservationWrapper):
    """This wrapper applies np.squeeze to make shape of observation compatible with stb3"""

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        observation = np.squeeze(observation)
        return observation


class RescaleWrapper(gym.ObservationWrapper):
    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        return observation / 255.0


class LidarWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: LidarConfig) -> None:
        super().__init__(env)
        self._lidar = Lidar(config)

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        return self._lidar.scan_2d(observation)[0]


class TrackSegmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: TrackSegmentationConfig) -> None:
        super().__init__(env)
        self._track_segmenter = TrackSegmenter(config)

    @log_observation(__name__)
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._track_segmenter.perform_segmentation(observation)
