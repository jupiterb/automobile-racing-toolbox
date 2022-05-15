import numpy as np

from enviroments.real.state.ocr.abstract import AbstractOcr
from schemas.game.feature_extraction.segment_detection_params import SegmentDetectionParams
from schemas import GameGlobalConfiguration, ScreenFrame


class SegmentDetectionOcr(AbstractOcr):

    def __init__(self, global_configuration: GameGlobalConfiguration, segment_detection_params: SegmentDetectionParams) -> None:
        super().__init__(global_configuration)
        self._params: SegmentDetectionParams = segment_detection_params

    @staticmethod
    def __get_segment(image: np.ndarray, segment_definition: ScreenFrame) -> np.ndarray:
        height, width = image.shape
        top, left, bottom, right = \
            int(height * segment_definition.top), int(width * segment_definition.left), \
            int(height * segment_definition.bottom), int(width * segment_definition.right)
        return image[top:bottom, left:right]

    def __is_segment_coveraged(self, segemnt: np.ndarray) -> bool:
        return segemnt.sum() / (segemnt.shape[0] * segemnt.shape[1] * 255) >= self._params.minimal_segment_coverage

    def __read_digit(self, image: np.ndarray) -> int:
        segments_coveraged: list[int] = [
            segment for segment, definition in self._params.segments_definitions.items() 
            if self.__is_segment_coveraged(self.__get_segment(image, definition))
        ]
        potentail_digitis = [
            digit for digit, definition in self._params.digits_definitions.items() 
            if set(definition) == set(segments_coveraged)
        ]
        if any(potentail_digitis):
            return potentail_digitis[0]
        else:
            return 0 

    def read_number(self, image: np.ndarray) -> int:
        digits_captured = self._separated_digits(self._prepare_image(image))
        velocity = 0
        for digit in digits_captured:
            velocity = velocity * 10 + self.__read_digit(digit)
        return velocity
