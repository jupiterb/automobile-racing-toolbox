from typing import Type

from interface.interface import GameInterface, FramedOcr
from interface.models import GameConfiguration
from interface.ocr import SevenSegmentsOcr
from interface.capturing import GameActionCapturing, KeyboardCapturing, GamepadCapturing
from interface.controllers import (
    GameActionController,
    KeyboardController,
    GamepadController,
)
from interface.screen import LocalScreen


def from_config(
    config: GameConfiguration,
    controller_type: Type[GameActionController] | None = None,
    capturing_type: Type[GameActionCapturing] | None = None,
) -> GameInterface:
    screen = LocalScreen(
        config.process_name,
        config.window_size,
        config.obervation_frame,
    )

    ocrs = [
        FramedOcr(name, frame, SevenSegmentsOcr(ocr_config))
        for name, (frame, ocr_config) in config.ocrs.items()
    ]

    controller = None
    if controller_type == KeyboardController:
        controller = KeyboardController(
            config.discrete_actions_mapping,
            config.reset_keys_sequence,
        )
    elif controller_type == GamepadController:
        controller = GamepadController(
            config.continous_actions_mapping,
            config.reset_gamepad_sequence,
        )

    capturing = None
    if capturing_type == KeyboardCapturing:
        key_to_action_mapping = {
            key: action for action, key in config.discrete_actions_mapping.items()
        }
        capturing = KeyboardCapturing(key_to_action_mapping)
    elif capturing_type == GamepadCapturing:
        gamepad_to_action_mapping = {
            gamepad_action: action
            for action, gamepad_action in config.continous_actions_mapping.items()
        }
        capturing = GamepadCapturing(gamepad_to_action_mapping)

    return GameInterface(
        config.game_id, config.reset_seconds, screen, controller, capturing, ocrs
    )
