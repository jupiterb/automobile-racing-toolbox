import numpy as np
import cv2 

from schemas.game.feature_extraction import SegmentDetectionParams, OcrType
from schemas import State, GameGlobalConfiguration
from enviroments.real.state.ocr import AbstractOcr, SegmentDetectionOcr


class RealStateBuilder:
    def __init__(self, global_configuration: GameGlobalConfiguration) -> None:
        self._state = State()
        self._observation_shape = global_configuration.observation_shape
        ocr_velocity_params = global_configuration.ocr_velocity_params
        if ocr_velocity_params.ocr_type is OcrType.SEGMENT_DETECTION and ocr_velocity_params.segment_detection_params:
            self._ocr: AbstractOcr = SegmentDetectionOcr(global_configuration, ocr_velocity_params.segment_detection_params)
        else:
             self._ocr: AbstractOcr = SegmentDetectionOcr(global_configuration, SegmentDetectionParams())

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
        self._state.velocity = self._ocr.read_number(screenshot)

    def build(self) -> State:
        result = self._state
        self.reset()
        return result
