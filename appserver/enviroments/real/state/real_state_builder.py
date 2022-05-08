import numpy as np
import cv2 
from schemas import State

class RealStateBuilder:
    def __init__(self, observation_shape) -> None:
        self._state = State()
        self._observation_shape = observation_shape

    def reset(self):
        self._state = State()

    def add_features_from_screenshot(self, screenshot: np.ndarray):
        grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        w, h, *_ = self._observation_shape
        resized = cv2.resize(grayscale, (w, h), cv2.INTER_AREA)
        self._state.screenshot_numpy_array = np.array(resized, dtype=np.uint8)
        if len(self._observation_shape) == 3 and self._observation_shape[-1] == 1:
            self._state.screenshot_numpy_array = np.array(resized, dtype=np.uint8)[:, :, None] # adds single channel

    def add_velocity_with_ocr(self, screenshot: np.ndarray):
        pass

    def get_result(self) -> State:
        result = self._state
        self.reset()
        return result 
