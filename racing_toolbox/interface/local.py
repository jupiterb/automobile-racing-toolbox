import numpy as np

from interface import GameInterface
from interface.models import GameConfiguration, SteeringAction
from interface.components import Keyboard, Screen
from interface.ocr import SevenSegmentsOcr
from pynput.keyboard import Controller, Listener, Key
import time

class LocalGameInterface(GameInterface):
    def __init__(self, configuration: GameConfiguration) -> None:
        super().__init__()
        self._configuration = configuration
        self._keyboard = Controller()#Keyboard(set(configuration.discrete_actions_mapping.values()))
        
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

    def name(self) -> str:
        return self._configuration.game_id

    def reset(self) -> None:
        """Method responsible for resseting enviroment"""
        self._keyboard.press(Key.enter)
        self._keyboard.release(Key.enter)
        time.sleep(3)


    def grab_image(self) -> np.ndarray:
        self._last_image = self._screen.grab_image(self._configuration.obervation_frame)
        return self._last_image

    def perform_ocr(self, on_last_image: bool = True) -> dict[str, float]:
        return {
            name: ocr.read_numer(self._screen.grab_image(frame, on_last_image))
            for name, frame, ocr in self._ocrs
        }

    def apply_action(self, discrete_actions: list[SteeringAction]) -> None:
        for a in discrete_actions:
            self._keyboard.press(self._configuration.discrete_actions_mapping[a])
        for a in set(SteeringAction) - set(discrete_actions):
            self._keyboard.release(self._configuration.discrete_actions_mapping[a])

    def read_action(self) -> list[SteeringAction]:
        return [self._keys_mapping[key] for key in self._keyboard.get_pressed()]
