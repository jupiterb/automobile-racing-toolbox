from pydantic import BaseModel, validator
from pynput.keyboard import Key
from typing import Any, Union
from racing_toolbox.interface.models.gamepad_action import GamepadAction
from racing_toolbox.observation.utils.ocr import OcrToolConfiguration


class GameConfiguration(BaseModel):
    game_id: str
    process_name: str
    window_size: tuple[int, int]
    discrete_actions_mapping: dict[str, Union[Key, str]]
    continous_actions_mapping: dict[str, GamepadAction]
    ocrs: OcrToolConfiguration
    reset_seconds: int
    reset_keys_sequence: list[Union[Key, str]]
    reset_gamepad_sequence: list[GamepadAction]
    frequency_per_second: int

    @validator("discrete_actions_mapping", pre=True, allow_reuse=True)
    def convert_discrete_mapping(cls, raw):
        assert isinstance(raw, dict), "not a dict"
        return {str(v): cls.__convert_key(k) for v, k in raw.items()}

    @validator("reset_keys_sequence", pre=True, each_item=True, allow_reuse=True)
    def convert_reset_keys_sequence(cls, raw):
        return cls.__convert_key(raw)

    @staticmethod
    def __convert_key(k: Any) -> Key:
        return k if isinstance(k, Key) else Key[str(k)]

    class Config:
        # use_enum_values = True
        json_encoders = {Key: lambda k: k.name, GamepadAction: lambda a: a.name}
