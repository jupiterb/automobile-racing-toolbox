from pydantic import BaseModel, PositiveInt
from typing import Optional
from racing_toolbox.observation.config import VAEConfig

from racing_toolbox.observation.utils import ScreenFrame


class ObservationConfig(BaseModel):
    frame: ScreenFrame
    shape: tuple[PositiveInt, PositiveInt]
    stack_size: PositiveInt
    use_lidar: bool = False
    vae_config: Optional[VAEConfig]
    observe_speed: bool = False
