import numpy as np

from interface import GameInterface
from interface.models import GameConfiguration, SteeringAction, ControllerType
from interface.capturing import GameActionCapturing, KeyboardCapturing, GamepadCapturing
from interface.controllers import (
    GameActionController,
    KeyboardController,
    GamepadController,
)
from interface.screen import Screen
from interface.ocr import SevenSegmentsOcr
import time


class LocalGameInterface(GameInterface):
    def __init__(
        self, configuration: GameConfiguration, controller_type: ControllerType
    ) -> None:
        super().__init__()
        self._configuration = configuration
        self._screen = Screen(configuration.process_name, configuration.window_size)
        self._ocrs = [
            (name, frame, SevenSegmentsOcr(ocr_configuration))
            for name, (frame, ocr_configuration) in configuration.ocrs.items()
        ]
        self._enable_action_read: bool = True

        if controller_type == ControllerType.KEYBOARD:
            self._controller: GameActionController = KeyboardController(
                configuration.discrete_actions_mapping,
                configuration.reset_keys_sequence,
            )
            key_to_action_mapping = {
                key: action
                for action, key in configuration.discrete_actions_mapping.items()
            }
            self._capturing: GameActionCapturing = KeyboardCapturing(
                key_to_action_mapping
            )

        else:
            self._controller: GameActionController = GamepadController(
                configuration.continous_actions_mapping,
                configuration.reset_gamepad_sequence,
            )
            gamepad_to_action_mapping = {
                gamepad_action: action
                for action, gamepad_action in configuration.continous_actions_mapping.items()
            }
            self._capturing: GameActionCapturing = GamepadCapturing(
                gamepad_to_action_mapping
            )

    def name(self) -> str:
        return self._configuration.game_id

    def reset(self, enable_action_read: bool = True) -> None:
        self._controller.reset_game()
        self._capturing.stop()
        time.sleep(self._configuration.reset_seconds)
        if enable_action_read:
            self._capturing.start()

    def grab_image(self) -> np.ndarray:
        return self._screen.grab_image(self._configuration.obervation_frame)

    def perform_ocr(self, on_last_image: bool = True) -> dict[str, float]:
        return {
            name: ocr.read_numer(self._screen.grab_image(frame, on_last_image))
            for name, frame, ocr in self._ocrs
        }

    def apply_action(self, actions: dict[SteeringAction, float]) -> None:
        self._controller.apply_actions(actions)

    def read_action(self) -> dict[SteeringAction, float]:
        return self._capturing.get_captured()
