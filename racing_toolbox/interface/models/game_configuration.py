from pydantic import BaseModel
from pynput.keyboard import Key
from enum import Enum

from interface.models.screen_frame import ScreenFrame
from interface.models.steereing_actions import SteeringAction
from interface.models.ocr_configuration import OcrConfiguration
from interface.models.gamepad_action import GamepadAction


class GameConfiguration(BaseModel):
    game_id: str
    process_name: str
    window_size: tuple[int, int]
    obervation_frame: ScreenFrame
    discrete_actions_mapping: dict[SteeringAction, Key]
    continous_actions_mapping: dict[SteeringAction, GamepadAction]
    ocrs: dict[str, tuple[ScreenFrame, OcrConfiguration]]
    reset_seconds: int
    reset_keys_sequence: list[Key]
    reset_gamepad_sequence: list[GamepadAction]
    frequency_per_second: int
    