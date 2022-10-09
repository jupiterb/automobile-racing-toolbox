import numpy as np
import cv2

from racing_toolbox.observation.config import LidarConfig, TrackSegmentationConfig
from racing_toolbox.observation import Lidar, TrackSegmenter
from tests import TEST_DIR


class LidarWithTrackSegmentation(Lidar):
    def __init__(self, lidar_config: LidarConfig, segmenter_config) -> None:
        super().__init__(lidar_config)
        self._segmenter = TrackSegmenter(segmenter_config)

    def scan_2d(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = self._segmenter.perform_segmentation(image)
        return super().scan_2d(image)


lidar = LidarWithTrackSegmentation(
    LidarConfig(
        depth=3,
        angles_range=(-90, 90, 10),
        lidar_start=(0.9, 0.5),
    ),
    TrackSegmentationConfig(
        track_color=(200, 200, 200),
        tolerance=80,
        noise_reduction=5,
    ),
)


def test_shape_of_lidar_result() -> None:
    distances, cooridnates = lidar.scan_2d(np.zeros((10, 10, 3), dtype=np.uint8))
    assert distances.shape == (19, 3)
    assert cooridnates.shape == (19, 3, 2)


def test_values_of_lidar_result() -> None:
    image = np.full((100, 100, 3), 150, dtype=np.uint8)
    image[:, :20, :] = 0
    image[:45, :, 1] = 0

    distances, cooridnates = lidar.scan_2d(image)

    x, y = cooridnates[0, 0]
    assert (x, y) == (90, 20)
    assert distances[0, 0] == 0.6

    x, y = cooridnates[9, 0]
    assert (x, y) == (45, 50)
    assert distances[9, 0] == 0.5

    x, y = cooridnates[-1, 0]
    assert (x, y) == (90, 99)
    assert distances[-1, 0] == 1.0

    for distance in distances[:, 1]:
        assert distance == 1.0

    for distance in distances[:, 2]:
        assert distance == 1.0

    image[:, 5:15, :] = 150
    distances, cooridnates = lidar.scan_2d(image)

    assert distances[0, 0] == 0.6
    assert distances[0, 1] == 0.9
    assert distances[0, 2] == 1.0


def test_lidar_with_real_images() -> None:
    in_the_middle = cv2.imread(
        str(TEST_DIR / "assets/screenshots/cropped/car_in_the_middle.jpeg")
    )
    on_the_left = cv2.imread(
        str(TEST_DIR / "assets/screenshots/cropped/car_on_the_left.jpeg")
    )
    on_the_edge = cv2.imread(
        str(TEST_DIR / "assets/screenshots/cropped/car_on_the_edge.jpeg")
    )

    distances_from_the_middle, _ = lidar.scan_2d(in_the_middle)
    distances_from_the_left, _ = lidar.scan_2d(on_the_left)
    distances_on_the_edge, _ = lidar.scan_2d(on_the_edge)

    for i in range(0, 4):  # most left lidars
        assert distances_from_the_left[i, 0] < distances_from_the_middle[i, 0]

    for i in range(0, 4):  # most left lidars
        assert distances_on_the_edge[i, 0] < 1 - 0.15 * i

    for i in range(-1, -4, -1):  # most right lidars,
        assert distances_on_the_edge[i, 0] < 0.2
        assert distances_on_the_edge[i, 1] == 1.0
        assert distances_on_the_edge[i, 2] == 1.0
