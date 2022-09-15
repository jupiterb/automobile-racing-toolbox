from pydantic import BaseModel
from pynput.keyboard import Key
from typing import Optional
from interface.models.screen_frame import ScreenFrame
from interface.models.steereing_actions import SteeringAction


class OcrConfiguration(BaseModel):
    threshold: int
    max_digits: int
    segemnts_definitions: Optional[dict[int, ScreenFrame]]


class GameConfiguration(BaseModel):
    game_id: str
    process_name: str
    window_size: tuple[int, int]
    observation_frame: ScreenFrame
    discrete_actions_mapping: dict[SteeringAction, Key]
    ocrs: dict[str, tuple[ScreenFrame, OcrConfiguration]]
    reset_seconds: int
    reset_keys_sequence: list[Key]
    frequency_per_second: int
