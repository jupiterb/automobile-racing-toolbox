from pydantic import BaseModel
from schemas.utils import ScreenFrame


class GameSystemConfiguration(BaseModel):
   velocity_screen_frame: ScreenFrame = ScreenFrame()
   driving_screen_frame: ScreenFrame = ScreenFrame()
