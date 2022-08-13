import numpy as np

from interface import GameInterface
from interface.models import GameConfiguration, SteeringAction
from interface.components import KeyboardController, KeyboardCapturing, Screen
from interface.ocr import SevenSegmentsOcr
import time


class LocalGameInterface(GameInterface):
    def __init__(self, configuration: GameConfiguration) -> None:
        super().__init__()
        self._configuration = configuration

        self._available_keys = set(configuration.discrete_actions_mapping.values())
        self._keyboard_controller = KeyboardController(self._available_keys)

        self._screen = Screen(configuration.process_name, configuration.window_size)

        self._ocrs = [
            (name, frame, SevenSegmentsOcr(ocr_configuration))
            for name, (frame, ocr_configuration) in configuration.ocrs.items()
        ]

        self._keyboard_capturing = KeyboardCapturing(self._available_keys)
        self._keys_mapping = {
            key: action
            for action, key in self._configuration.discrete_actions_mapping.items()
        }

        self._enable_action_read: bool = True

    def name(self) -> str:
        return self._configuration.game_id

    def reset(self) -> None:
        for key in self._configuration.reset_keys_sequence:
            self._keyboard_controller.click(key)
        time.sleep(self._configuration.reset_seconds)
        self._keyboard_capturing.stop()
        self._keyboard_capturing.start()

    def enable_action_read(self, enable: bool) -> None:
        if self._enable_action_read != enable:
            if enable:
                self._keyboard_capturing.start()
            else:
                self._keyboard_capturing.stop()
        self._enable_action_read = enable

    def grab_image(self) -> np.ndarray:
        return self._screen.grab_image(self._configuration.obervation_frame)

    def perform_ocr(self, on_last_image: bool = True) -> dict[str, float]:
        return {
            name: ocr.read_numer(self._screen.grab_image(frame, on_last_image))
            for name, frame, ocr in self._ocrs
        }

    def apply_action(self, discrete_actions: dict[SteeringAction, float]) -> None:
        keys = [
            self._configuration.discrete_actions_mapping[action]
            for action in discrete_actions
        ]
        self._keyboard_controller.set_pressed(keys)

    def read_action(self) -> dict[SteeringAction, float]:
        return {
            self._keys_mapping[key]: 1 for key in self._keyboard_capturing.get_pressed()
        }
