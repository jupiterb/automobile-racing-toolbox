from pydantic import BaseModel, PositiveInt
from observation.config import LidarConfig, TrackSegmentationConfig


class ObservationConfig(BaseModel):
    shape: tuple[PositiveInt, PositiveInt]
    stack_size: PositiveInt
    lidar_config: LidarConfig | None
    track_segmentation_config: TrackSegmentationConfig | None
