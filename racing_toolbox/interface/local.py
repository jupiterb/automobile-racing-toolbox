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
            (name, frame, SevenSegmentsOcr(ocr_configuration))
            for name, (frame, ocr_configuration) in configuration.ocrs.items()
        ]
        self._last_image: np.ndarray = np.zeros_like(configuration.window_size)

    def reset(self) -> None:
        self._keyboard.reset()

    def grab_image(self) -> np.ndarray:
        self._last_image = self._screen.grab_image(self._configuration.obervation_frame)
        return self._last_image

    def perform_ocr(self, on_last_image: bool = True) -> dict[str, int]:
        return {
            name: ocr.read_numer(self._screen.grab_image(frame, on_last_image))
            for name, frame, ocr in self._ocrs
        }

    def apply_action(self, discrete_actions: list[SteeringAction]) -> None:
        keys = [
            self._configuration.discrete_actions_mapping[action]
            for action in discrete_actions
        ]
        self._keyboard.set_pressed(keys)

    def read_action(self) -> list[SteeringAction]:
        return [self._keys_mapping[key] for key in self._keyboard.get_pressed()]
