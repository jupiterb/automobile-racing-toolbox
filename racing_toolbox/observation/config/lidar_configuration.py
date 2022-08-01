from pydantic import BaseModel


class LidarConfig(BaseModel):
    threshold: int
    kernel_size: int
    lidar_start: tuple[float, float]
    rays_angles_range: tuple[int, int, int]
