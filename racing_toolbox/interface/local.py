import numpy as np

from interface import GameInterface
from interface.models import GameConfiguration, SteeringAction
from interface.components import Keyboard, ScreenCapturing
from interface.ocr import SevenSegmentsOcr


class LocalGameInterface(GameInterface):
    def __init__(self, configuration: GameConfiguration) -> None:
        super().__init__()
        self._configuration = configuration
        self._ocr = SevenSegmentsOcr()
        self._keyboard = Keyboard(set(configuration.discrete_actions_mapping.values()))
        self._screen = ScreenCapturing(
            configuration.process_name, configuration.window_size
        )
        self._keys_mapping = {
            key: action
            for action, key in self._configuration.discrete_actions_mapping.items()
        }

    def restart(self) -> None:
        self._keyboard.restart()

    def grab_image(self) -> np.ndarray:
        return self._screen.grab_image(self._configuration.obervation_frame)

    def perform_ocr(self) -> list[int]:
        return [
            self._ocr.read_numer(self._screen.grab_image(frame))
            for frame in self._configuration.ocr_frames
        ]

    def apply_action(self, discrete_actions: list[SteeringAction]) -> None:
        keys = [
            self._configuration.discrete_actions_mapping[action]
            for action in discrete_actions
        ]
        self._keyboard.set_pressed(keys)

    def read_action(self) -> list[SteeringAction]:
        return [self._keys_mapping[key] for key in self._keyboard.get_pressed()]
