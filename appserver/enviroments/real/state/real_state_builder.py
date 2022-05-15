import numpy as np

from schemas.game.feature_extraction import SegmentDetectionParams, OcrType
from schemas import State, GameGlobalConfiguration
from enviroments.real.state.ocr import AbstractOcr, SegmentDetectionOcr


class RealStateBuilder():

    def __init__(self, global_configuration: GameGlobalConfiguration) -> None:
        self._state = State()
        ocr_velocity_params = global_configuration.ocr_velocity_params
        if ocr_velocity_params.ocr_type is OcrType.SEGMENT_DETECTION and ocr_velocity_params.segment_detection_params:
            self._ocr: AbstractOcr = SegmentDetectionOcr(global_configuration, ocr_velocity_params.segment_detection_params)
        else:
             self._ocr: AbstractOcr = SegmentDetectionOcr(global_configuration, SegmentDetectionParams())

    def reset(self):
        self._state = State()

    def add_features_from_screenshot(self, screenshot: np.ndarray):
        self._state.screenshot_numpy_array = screenshot

    def add_velocity_with_ocr(self, screenshot: np.ndarray):
        self._state.velocity = self._ocr.read_number(screenshot)

    def get_result(self) -> State:
        return self._state
