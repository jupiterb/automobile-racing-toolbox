from typing import Any
import numpy as np
import cv2

from observation.config import LidarConfig


class Lidar:
    def __init__(self, config: LidarConfig) -> None:
        self._config = config
        self._shape: tuple[int, int] = (0, 0)
        self._distances: list[list[np.floating[Any]]] = []
        self._rays_coordinates: list[list[tuple[int, int]]] = []

    def scan_2d(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.shape != self._shape:
            self._shape = image.shape[0:2]
            self._set_rays()
        image = self._preprocess_image(image)
        collisions = [
            self._first_collision_with_edge(ray_coordinates, image)
            for ray_coordinates in self._rays_coordinates
        ]
        distances_to_collisions = np.array(
            [self._distances[i][col] for i, col in enumerate(collisions)]
        )
        cooridnates_of_collisions = np.array(
            [self._rays_coordinates[i][col] for i, col in enumerate(collisions)]
        )
        return distances_to_collisions, cooridnates_of_collisions

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim > 2 else image
        # gaussian blur
        size = self._config.kernel_size
        kernel = np.ones((size, size), np.float32)
        image = cv2.filter2D(image, -1, kernel / size**2)
        # find points with road
        road_color = image[self._get_start_point()]
        image[image > road_color + self._config.threshold] = 0
        image[image < road_color - self._config.threshold] = 0
        image[image > 0] = 1
        return image

    def _first_collision_with_edge(
        self, ray_coordinates: list[tuple[int, int]], image: np.ndarray
    ) -> int:
        off_track = np.nonzero([image[coords] == 0 for coords in ray_coordinates])[0]
        return off_track[0] if any(off_track) else len(ray_coordinates) - 1

    def _set_rays(self) -> None:
        start, end, angle_between = self._config.rays_angles_range
        end = end if end % angle_between else end + angle_between
        start_point = self._get_start_point()
        self._distances = []
        self._rays_coordinates = []
        for angle in range(start, end, angle_between):
            self._add_ray(start_point, angle)

    def _add_ray(self, start_point: tuple[int, int], angle: int) -> None:
        """
        the angles go clockwise, including the zero angle is vertical
        """
        dir_factors = (-np.cos(np.radians(angle)), np.sin(np.radians(angle)))

        distances: list[np.floating[Any]] = []
        coordinates = []

        iter = 0
        next_point_of_ray = Lidar._get_point_of_ray(iter, start_point, dir_factors)
        while self._is_point_on_observation(next_point_of_ray):
            iter += 1
            coordinates.append(next_point_of_ray)
            distances.append(np.linalg.norm(np.array(start_point) - next_point_of_ray))
            next_point_of_ray = Lidar._get_point_of_ray(iter, start_point, dir_factors)

        self._distances.append(distances)
        self._rays_coordinates.append(coordinates)

    def _is_point_on_observation(self, point: tuple[int, int]) -> bool:
        ray_x, ray_y = point
        height, width = self._shape
        return 0 <= ray_x < height and 0 <= ray_y < width

    def _get_start_point(self) -> tuple[int, int]:
        x, y = np.array(self._config.lidar_start) * self._shape
        return int(x), int(y)

    @staticmethod
    def _get_point_of_ray(
        index: int,
        start_point: tuple[int, int],
        dir_factors: tuple[float, float],
    ) -> tuple[int, int]:
        return (
            int(start_point[0] + index * dir_factors[0]),
            int(start_point[1] + index * dir_factors[1]),
        )
