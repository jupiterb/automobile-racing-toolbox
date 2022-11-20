from pydantic import BaseModel
from typing import Optional
from racing_toolbox.observation.utils.screen_frame import ScreenFrame


class OcrConfiguration(BaseModel):
    threshold: int
    max_digits: int
    segemnts_definitions: Optional[dict[int, ScreenFrame]]


class OcrToolConfiguration(BaseModel):
    instances: dict[str, tuple[ScreenFrame, OcrConfiguration]]
