from pydantic import BaseModel
from typing import Optional
from schemas.utils import ScreenFrame


class GameSystemConfiguration(BaseModel):
   velocity_screen_frame: Optional[ScreenFrame] = None
   driving_screen_frame: Optional[ScreenFrame] = None
