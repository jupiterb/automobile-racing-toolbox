import sys
from os import path
import numpy as np
import cv2

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from observation.config import LidarConfig
from observation import Lidar


config = LidarConfig(
    depth=3,
    lower_threshold=60,
    upper_threshold=255,
    kernel_size=5,
    angles_range=(-90, 90, 10),
    lidar_start=(0.9, 0.5),
)
lidar = Lidar(config)


def test_shape_of_lidar_result() -> None:
    distances, cooridnates = lidar.scan_2d(np.zeros((10, 10), dtype=np.uint8))
    assert distances.shape == (19, 3)
    assert cooridnates.shape == (19, 3, 2)

    distances, cooridnates = lidar.scan_2d(np.zeros((10, 10, 3), dtype=np.uint8))
    assert distances.shape == (19, 3)
    assert cooridnates.shape == (19, 3, 2)


def test_values_of_lidar_result() -> None:
    image = np.full((100, 100), 75, dtype=np.uint8)
    image[:, :20] = 0
    image[:45, :] = 0

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

    image[:, 5:15] = 75
    distances, cooridnates = lidar.scan_2d(image)

    assert distances[0, 0] == 0.6
    assert distances[0, 2] == 1.0
    assert distances[0, 0] < distances[0, 1] < distances[0, 2]


def test_lidar_with_real_images() -> None:
    in_the_middle = cv2.imread("assets/screenshots/cropped/car_in_the_middle.jpeg")
    on_the_left = cv2.imread("assets/screenshots/cropped/car_on_the_left.jpeg")
    on_the_edge = cv2.imread("assets/screenshots/cropped/car_on_the_edge.jpeg")

    distances_from_the_middle, _ = lidar.scan_2d(in_the_middle)
    distances_from_the_left, _ = lidar.scan_2d(on_the_left)
    distances_on_the_edge, _ = lidar.scan_2d(on_the_edge)

    for i in range(0, 4):  # most left lidars
        assert distances_from_the_left[i, 0] < distances_from_the_middle[i, 0]

    for i in range(0, 4):  # most left lidars
        assert distances_on_the_edge[i, 0] < 1 - 0.15 * i

    for i in range(-1, -5, -1):  # most right lidars,
        assert distances_on_the_edge[i, 0] < 0.2
        assert distances_on_the_edge[i, 1] > 0.95
        assert distances_on_the_edge[i, 2] == 1.0
