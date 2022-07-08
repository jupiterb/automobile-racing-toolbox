import numpy as np

from interface import GameInterface
from interface.models import GameConfiguration, SteeringAction
from interface.components import Keyboard, Screen
from interface.ocr import SevenSegmentsOcr


class LocalGameInterface(GameInterface):
    def __init__(self, configuration: GameConfiguration) -> None:
        super().__init__()
        self._configuration = configuration
        self._keyboard = Keyboard(set(configuration.discrete_actions_mapping.values()))
        self._screen = Screen(configuration.process_name, configuration.window_size)
        self._keys_mapping = {
            key: action
            for action, key in self._configuration.discrete_actions_mapping.items()
        }
        self._ocrs = [
            SevenSegmentsOcr(ocr_configuration)
            for _, ocr_configuration in configuration.ocrs
        ]
        self._last_image: np.ndarray = np.zeros_like(configuration.window_size)

    def restart(self) -> None:
        self._keyboard.restart()

    def grab_image(self) -> np.ndarray:
        self._last_image = self._screen.grab_image(self._configuration.obervation_frame)
        return self._last_image

    def perform_ocr(self, on_last_image: bool = True) -> list[int]:
        return [
            ocr.read_numer(self._screen.grab_image(frame, on_last_image))
            for ocr, (frame, _) in zip(self._ocrs, self._configuration.ocrs)
        ]

    def apply_action(self, discrete_actions: list[SteeringAction]) -> None:
        keys = [
            self._configuration.discrete_actions_mapping[action]
            for action in discrete_actions
        ]
        self._keyboard.set_pressed(keys)

    def read_action(self) -> list[SteeringAction]:
        return [self._keys_mapping[key] for key in self._keyboard.get_pressed()]
