from pydantic import BaseModel, PositiveInt
from typing import Optional
from racing_toolbox.observation.config import (
    LidarConfig,
    TrackSegmentationConfig,
    VAEConfig,
)
from racing_toolbox.observation.utils import ScreenFrame


class ObservationConfig(BaseModel):
    frame: ScreenFrame
    shape: tuple[PositiveInt, PositiveInt]
    stack_size: PositiveInt
    lidar_config: Optional[LidarConfig] = None
    track_segmentation_config: Optional[TrackSegmentationConfig] = None
    vae_config: Optional[VAEConfig]
    observe_speed: bool = False
