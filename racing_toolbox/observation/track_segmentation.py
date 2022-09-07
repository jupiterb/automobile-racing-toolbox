import cv2
import numpy as np

from observation.config import TrackSegmentationConfig

def cross(h, w):
    mask = np.zeros((h, h))
    mid = h // 2
    mask[mid - w // 2: mid + w//2, :] = 1
    mask[:, mid - w//2 : mid + w//2] = 1
    return mask.astype(np.uint8) 

def star(h):
    mask = np.zeros((h, h))
    mask = np.eye(h) + np.rot90(np.eye(h), 2)
    mask[mask > 0] = 1
    return mask.astype(np.uint8)

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

    def _perform_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        return 2-dimensional array, wehere if cell is equal to 0, there is no track
        """
        assert image.ndim == 3 and image.shape[2] == 3

        size = self._config.noise_reduction
        # gaussian filter
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2]
        image = cv2.GaussianBlur(image, (size, size), 10)
        lower = self._config.track_color - self._config.tolerance
        upper = self._config.track_color + self._config.tolerance
        image = cv2.inRange(image, lower, upper)
        # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5, 5)))
        # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cross(11, 4))
        # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, star(10))
        return image
