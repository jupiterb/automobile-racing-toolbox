import numpy as np

from schemas import State

class RealStateBuilder():

    def __init__(self) -> None:
        self._state = State()

    def reset(self):
        self._state = State()

    def add_features_from_screenshot(self, screenshot: np.ndarray):
        pass

    def add_velocity_with_ocr(self, screenshot: np.ndarray):
        pass

    def get_result(self) -> State:
        return self._state
