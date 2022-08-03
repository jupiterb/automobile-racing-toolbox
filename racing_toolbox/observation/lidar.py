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
        """
        Returns tuple of:
            1. distances (np.ndarray of shape (lidars_count, depth))
                to first <depth> elements on which lidars had a collision
            2. coordinates (np.ndarray of shape (lidars_count, depth, 2))
                of first <depth> elements on which lidars had a collision
        """
        if image.shape != self._shape:
            self._shape = image.shape[0:2]
            self._set_lidars()
        image = self._preprocess_image(image)
        collisions = [
            self._get_collisions(ray_coordinates, image)
            for ray_coordinates in self._rays_coordinates
        ]
        distances_to_collisions = np.array(
            [
                [
                    self._distances[i][j] / self._distances[i][-1]
                    for j in collisions_on_lidar
                ]
                for i, collisions_on_lidar in enumerate(collisions)
            ]
        )
        coordinates_of_collisions = np.array(
            [
                [self._rays_coordinates[i][j] for j in collisions_on_lidar]
                for i, collisions_on_lidar in enumerate(collisions)
            ]
        )
        return distances_to_collisions, coordinates_of_collisions

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        checks wheter point of image is on or off track
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

    def _get_collisions(
        self, ray_coordinates: list[tuple[int, int]], image: np.ndarray
    ) -> list[int]:
        """
        after collision with off track element
        it traets this element as on track and finds next one
        """
        off_track = np.nonzero([image[coords] == 0 for coords in ray_coordinates])[0]
        result = []
        for _ in range(self._config.depth):
            index = off_track[0] if any(off_track) else len(ray_coordinates) - 1
            result.append(index)
            while len(off_track) > 1 and off_track[0] == off_track[1] - 1:
                off_track = off_track[1:]
            off_track = off_track[1:] if any(off_track) else off_track
        return result

    def _set_lidars(self) -> None:
        """
        counts cooridnates of every point of lidars and distances to this points
        """
        start, end, angle_between = self._config.angles_range
        end = end if (end - start) % angle_between else end + angle_between
        start_point = self._get_start_point()
        self._distances = []
        self._rays_coordinates = []
        for angle in range(start, end, angle_between):
            self._add_lidar(start_point, angle)

    def _add_lidar(self, start_point: tuple[int, int], angle: int) -> None:
        """
        the angles go clockwise, including the zero angle is vertical
        """
        dir_factors = (-np.cos(np.radians(angle)), np.sin(np.radians(angle)))

        distances: list[np.floating[Any]] = []
        coordinates = []

        iter = 0
        next_point_of_ray = Lidar._get_lidar_point(iter, start_point, dir_factors)
        while self._is_point_on_observation(next_point_of_ray):
            iter += 1
            coordinates.append(next_point_of_ray)
            distances.append(np.linalg.norm(np.array(start_point) - next_point_of_ray))
            next_point_of_ray = Lidar._get_lidar_point(iter, start_point, dir_factors)

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
    def _get_lidar_point(
        index: int,
        start_point: tuple[int, int],
        dir_factors: tuple[float, float],
    ) -> tuple[int, int]:
        return (
            int(start_point[0] + index * dir_factors[0]),
            int(start_point[1] + index * dir_factors[1]),
        )
