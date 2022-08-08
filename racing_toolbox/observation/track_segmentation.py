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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim > 2 else image
        # gaussian blur
        size = self._config.kernel_size
        kernel = np.ones((size, size), np.float32)
        image = cv2.filter2D(image, -1, kernel / size**2)
        # find points with road
        image[image > self._config.upper_threshold] = 0
        image[image < self._config.lower_threshold] = 0
        image[image > 0] = 1
        # another noise reduction
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image
