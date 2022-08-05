"""
Parameters:
    1. path to the folder with images on which to test lidar
    2. path to destination folder. should don't exists.
"""

from PIL import Image
import sys
from os import path, listdir, makedirs
import cv2
import numpy as np

from observation import Lidar
from observation.config import LidarConfig


def perfrom_lidar_scaning(path_to_images: str, path_to_result: str) -> None:

    fullpath = f"{path.dirname(__file__)}/{path_to_result}"
    if path.exists(fullpath):
        print(f"{fullpath} exists.")
        return
    makedirs(fullpath)

    names = [name for name in listdir(path_to_images) if name[-5:] == ".jpeg"]
    images = [cv2.imread(f"{path_to_images}/{image_name}") for image_name in names]

    config = LidarConfig(
        depth=3,
        lower_threshold=60,
        upper_threshold=255,
        kernel_size=5,
        angles_range=(-90, 90, 10),
        lidar_start=(0.9, 0.5),
    )
    lidar = Lidar(config)

    for name, image in zip(names, images):
        Image.fromarray(scan_image(lidar, image)).save(f"{fullpath}/{name}")


def scan_image(lidar: Lidar, image: np.ndarray) -> np.ndarray:
    _, all_lidars_collision_points = lidar.scan_2d(image)
    start_point = lidar._get_start_point()
    for collision_points in all_lidars_collision_points:
        collision_points = list(collision_points)
        collision_points.reverse()
        color = [255, 0, 0]
        color_change = int(255 / len(collision_points))
        for point in collision_points:
            image = cv2.line(
                image, (start_point[1], start_point[0]), (point[1], point[0]), color, 3
            )
            color[0] -= color_change
            color[1] += color_change
    return image


if __name__ == "__main__":
    path_to_images = sys.argv[1]
    path_to_result = sys.argv[2]
    perfrom_lidar_scaning(path_to_images, path_to_result)
