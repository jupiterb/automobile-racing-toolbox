from interface.interface import GameInterface
from interface.models import GameConfiguration
from interface.ocr import SevenSegmentsOcr, OcrWrapper
from interface.capturing import KeyboardCapturing, GamepadCapturing
from interface.controllers import KeyboardController, GamepadController
from interface.screen import LocalScreen


class GameInterfaceBuilder:

    _config: GameConfiguration
    _interface: GameInterface

    def new_interface(self, configuration: GameConfiguration) -> None:
        self._config = configuration
        screen = LocalScreen(
            configuration.process_name,
            configuration.window_size,
            configuration.obervation_frame,
        )
        self._interface = GameInterface(
            configuration.game_id, screen, configuration.reset_seconds
        )
        ocrs = [
            OcrWrapper(frame, name, SevenSegmentsOcr(ocr_configuration))
            for name, (frame, ocr_configuration) in configuration.ocrs.items()
        ]
        self._interface.set_ocrs(ocrs)

    def build(self) -> GameInterface:
        return self._interface

    def with_gamepad_controller(self) -> None:
        controller = GamepadController(
            self._config.continous_actions_mapping,
            self._config.reset_gamepad_sequence,
        )
        self._interface.set_controller(controller)

    def with_keyborad_controller(self) -> None:
        controller = KeyboardController(
            self._config.discrete_actions_mapping,
            self._config.reset_keys_sequence,
        )
        self._interface.set_controller(controller)

    def with_gamepad_capturing(self) -> None:
        gamepad_to_action_mapping = {
            gamepad_action: action
            for action, gamepad_action in self._config.continous_actions_mapping.items()
        }
        capturing = GamepadCapturing(gamepad_to_action_mapping)
        self._interface.set_capturing(capturing)

    def with_keyboard_capturing(self) -> None:
        key_to_action_mapping = {
            key: action for action, key in self._config.discrete_actions_mapping.items()
        }
        capturing = KeyboardCapturing(key_to_action_mapping)
        self._interface.set_capturing(capturing)
