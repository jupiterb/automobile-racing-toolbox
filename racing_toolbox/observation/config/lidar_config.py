from pydantic import BaseModel


class LidarConfig(BaseModel):
    depth: int
    lidar_start: tuple[float, float]
    angles_range: tuple[int, int, int]
