from pydantic import BaseModel, PositiveInt
from observation.config import LidarConfig


class ObservationConfig(BaseModel):
    shape: tuple[PositiveInt, PositiveInt]
    stack_size: PositiveInt
    lidar_config: LidarConfig | None
