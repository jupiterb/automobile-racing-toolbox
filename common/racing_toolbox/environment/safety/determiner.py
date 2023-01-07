import numpy as np

from racing_toolbox.observation.config import LidarConfig, TrackSegmentationConfig
from racing_toolbox.observation.lidar import Lidar
from racing_toolbox.observation.track_segmentation import TrackSegmenter


class SafetyDeterminer:
    def __init__(
        self,
        lidar_config: LidarConfig,
        segmentation_config: TrackSegmentationConfig,
        shortest_rays_number: int,
        weight: float,
        centralization: float,
        lidar_depth: int,
    ) -> None:
        self._lidar = Lidar(lidar_config)
        self._segmenter = TrackSegmenter(segmentation_config)
        self._rays_number = shortest_rays_number
        self._weight = weight
        self._centralization = centralization
        self._lidar_depth = lidar_depth

    def safety(self, track_image: np.ndarray) -> float:
        segmented = self._segmenter.perform_segmentation(track_image)
        rays, _ = self._lidar.scan_2d(segmented)
        shortest_distances = sorted(rays[self._lidar_depth])[: self._rays_number]
        return np.mean(shortest_distances) ** self._centralization * self._weight
