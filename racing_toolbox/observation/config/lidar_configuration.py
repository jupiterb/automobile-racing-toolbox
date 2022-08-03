from pydantic import BaseModel


class LidarConfig(BaseModel):
    depth: int
    lower_threshold: int
    upper_threshold: int
    kernel_size: int
    lidar_start: tuple[float, float]
    angles_range: tuple[int, int, int]
