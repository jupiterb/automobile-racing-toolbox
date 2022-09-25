from pydantic import BaseModel
from pynput.keyboard import Key
from enum import Enum
from typing import Optional
from racing_toolbox.interface.models.screen_frame import ScreenFrame
from racing_toolbox.interface.models.gamepad_action import GamepadAction


class OcrConfiguration(BaseModel):
    threshold: int
    max_digits: int
    segemnts_definitions: Optional[dict[int, ScreenFrame]]


class GameConfiguration(BaseModel):
    game_id: str
    process_name: str
    window_size: tuple[int, int]
    observation_frame: ScreenFrame
    discrete_actions_mapping: dict[str, Key]
    continous_actions_mapping: dict[str, GamepadAction]
    ocrs: dict[str, tuple[ScreenFrame, OcrConfiguration]]
    reset_seconds: int
    reset_keys_sequence: list[Key]
    reset_gamepad_sequence: list[GamepadAction]
    frequency_per_second: int
