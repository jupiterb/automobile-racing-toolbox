from pydantic import BaseModel
from racing_toolbox.interface.models.screen_frame import ScreenFrame


class OcrConfiguration(BaseModel):
    threshold: int
    max_digits: int
    segemnts_definitions: dict[int, ScreenFrame] | None
