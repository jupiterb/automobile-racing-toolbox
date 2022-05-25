import numpy as np
import cv2
from schemas import State, GameGlobalConfiguration


class RealStateBuilder:
    def __init__(self, global_configuration: GameGlobalConfiguration) -> None:
        self._state = State()
        self._observation_shape = global_configuration.observation_shape

    def reset(self):
        self._state = State()

    def add_features_from_screenshot(self, screenshot: np.ndarray):
        grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) if screenshot.ndim > 2 else screenshot
        w, h, *_ = self._observation_shape
        resized = cv2.resize(grayscale, (w, h), cv2.INTER_AREA)
        self._state.screenshot_numpy_array = np.array(resized, dtype=np.uint8)
        if len(self._observation_shape) == 3 and self._observation_shape[-1] == 1:
            self._state.screenshot_numpy_array = np.array(resized, dtype=np.uint8)[
                :, :, None
            ]  # adds single channel

    def add_velocity(self, velocity: int):
        self._state.velocity = velocity

    def build(self) -> State:
        result = self._state
        self.reset()
        return result
