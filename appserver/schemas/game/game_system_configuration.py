from pydantic import BaseModel
from typing import Optional
from schemas.utils import ScreenFrame


class GameSystemConfiguration(BaseModel):
   velocity_screen_frame: ScreenFrame = ScreenFrame()
   driving_screen_frame: ScreenFrame = ScreenFrame()
   specified_window_rect: Optional[tuple[int, int, int, int]] = None # left, top, width, height
