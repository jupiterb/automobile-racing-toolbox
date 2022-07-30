from pydantic import BaseModel
from pynput.keyboard import Key

from interface.config.screen_frame import ScreenFrame
from interface.config.steereing_actions import SteeringAction
from interface.config.ocr_configuration import OcrConfiguration


class GameConfiguration(BaseModel):
    game_id: str
    process_name: str
    window_size: tuple[int, int]
    obervation_frame: ScreenFrame
    discrete_actions_mapping: dict[SteeringAction, Key]
    ocrs: dict[str, tuple[ScreenFrame, OcrConfiguration]]
    reset_seconds: int
    reset_keys_sequence: list[Key]
    frequency_per_second: int
    