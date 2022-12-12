from pydantic import BaseModel, PositiveInt
from typing import Optional
from racing_toolbox.observation.config import LidarConfig, TrackSegmentationConfig, VAEConfig
from racing_toolbox.observation.utils import ScreenFrame


class ObservationConfig(BaseModel):
    frame: ScreenFrame
    shape: tuple[PositiveInt, PositiveInt]
    stack_size: PositiveInt
    lidar_config: Optional[LidarConfig]
    track_segmentation_config: Optional[TrackSegmentationConfig]
    vae_config: Optional[VAEConfig]
