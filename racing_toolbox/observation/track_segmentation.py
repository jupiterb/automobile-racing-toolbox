import cv2
import numpy as np

from observation.config import TrackSegmentationConfig


class TrackSegmenter:
    def __init__(self, config: TrackSegmentationConfig) -> None:
        self._config = config

    def perform_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        return 2-dimensional array, wehere if cell is equal to 0, there is no track
        """
        assert image.ndim == 3 and image.shape[2] == 3
        # gaussian filter
        size = self._config.noise_reduction
        kernel = np.ones((size, size), np.float32)
        filtered = cv2.filter2D(image, -1, kernel / size**2)
        # find points with road
        lower = np.array(self._config.track_color) - self._config.tolerance
        upper = np.array(self._config.track_color) + self._config.tolerance
        mask = cv2.inRange(filtered, lower, upper)
        # another noise reduction
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return closed_mask
