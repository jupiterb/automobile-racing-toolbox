from pydantic import BaseModel
from interface.models.screen_frame import ScreenFrame
from typing import Optional

class OcrConfiguration(BaseModel):
    threshold: int
    max_digits: int
    segemnts_definitions: Optional[dict[int, ScreenFrame]]
