import gym
import numpy as np

from observation.config import LidarConfig, TrackSegmentationConfig

from observation.lidar import Lidar
from observation.track_segmentation import TrackSegmenter
import cv2
import gym.spaces


class SqueezingWrapper(gym.ObservationWrapper):
    """This wrapper applies np.squeeze to make shape of observation compatible with stb3"""

    def observation(self, observation: np.ndarray):
        observation = np.squeeze(observation)
        return observation


class RescaleWrapper(gym.ObservationWrapper):
    def observation(self, observation: np.ndarray):
        return observation / 255.0


class LidarWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: LidarConfig) -> None:
        super().__init__(env)
        self._lidar = Lidar(config)
        size = len(range(*config.angles_range)) + 1
        self.observation_space = gym.spaces.Box(0, 1, (size, config.depth), float)
        # self.cap = cv2.VideoCapture(0)

    def observation(self, observation: np.ndarray):
        # _, all_lidars_collision_points = self._lidar.scan_2d(observation)
        # start_point = self._lidar._get_start_point()
        # for collision_points in all_lidars_collision_points:
        #     collision_points = list(collision_points)
        #     collision_points.reverse()
        #     color = [255, 0, 0]
        #     color_change = int(255 / len(collision_points))
        #     for point in collision_points:
        #         observation = cv2.line(
        #             observation, (start_point[1], start_point[0]), (point[1], point[0]), color, 3
        #         )
        #         color[0] -= color_change
        #         color[1] += color_change
        # cv2.imshow('frame', observation)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     self.cap.release()
        #     cv2.destroyAllWindows()
        
        return self._lidar.scan_2d(observation)[0]


class TrackSegmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: TrackSegmentationConfig) -> None:
        super().__init__(env)
        self._track_segmenter = TrackSegmenter(config)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._track_segmenter.perform_segmentation(observation)
