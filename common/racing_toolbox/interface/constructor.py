from typing import Type, Optional

from racing_toolbox.interface.interface import GameInterface
from racing_toolbox.interface.config import GameConfiguration
import racing_toolbox.interface.capturing as capturing
import racing_toolbox.interface.controllers as controllers
import racing_toolbox.interface.screen as screen


def from_config(
    config: GameConfiguration,
    controller_type: Optional[Type[controllers.GameActionController]] = None,
    capturing_type: Optional[Type[capturing.GameActionCapturing]] = None,
) -> GameInterface:
    selected_screen = screen.LocalScreen(config.process_name, config.window_size)

    selected_controller = None
    if controller_type == controllers.KeyboardController:
        selected_controller = controllers.KeyboardController(
            config.discrete_actions_mapping,
            config.reset_keys_sequence,
        )
    elif controller_type == controllers.GamepadController:
        selected_controller = controllers.GamepadController(
            config.continous_actions_mapping,
            config.reset_gamepad_sequence,
        )

    selected_capturing = None
    if capturing_type == capturing.KeyboardCapturing:
        key_to_action_mapping = {
            key: action for action, key in config.discrete_actions_mapping.items()
        }
        selected_capturing = capturing.KeyboardCapturing(key_to_action_mapping)
    elif capturing_type == capturing.GamepadCapturing:
        gamepad_to_action_mapping = {
            gamepad_action: action
            for action, gamepad_action in config.continous_actions_mapping.items()
        }
        selected_capturing = capturing.GamepadCapturing(gamepad_to_action_mapping)

    return GameInterface(
        config.game_id,
        config.reset_seconds,
        selected_screen,
        selected_controller,
        selected_capturing,
    )
