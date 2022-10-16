from pydantic import BaseModel
from pynput.keyboard import Key
from racing_toolbox.interface.models.gamepad_action import GamepadAction
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.observation.utils.ocr import OcrConfiguration


class GameConfiguration(BaseModel):
    game_id: str
    process_name: str
    window_size: tuple[int, int]
    discrete_actions_mapping: dict[str, Key]
    continous_actions_mapping: dict[str, GamepadAction]
    ocrs: dict[str, tuple[ScreenFrame, OcrConfiguration]]
    reset_seconds: int
    reset_keys_sequence: list[Key]
    reset_gamepad_sequence: list[GamepadAction]
    frequency_per_second: int
