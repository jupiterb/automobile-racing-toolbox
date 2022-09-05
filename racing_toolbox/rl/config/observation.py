from pydantic import BaseModel, PositiveInt
from observation.config import LidarConfig, TrackSegmentationConfig
from typing import Optional 


class ObservationConfig(BaseModel):
    shape: tuple[PositiveInt, PositiveInt]
    stack_size: PositiveInt
    lidar_config: Optional[LidarConfig]
    track_segmentation_config: Optional[TrackSegmentationConfig]
