import numpy as np
import cv2

from interface.ocr.abstract import AbstractOcr
from interface.models import OcrConfiguration


class SevenSegmentsOcr(AbstractOcr):

    __segment_threshold: float = 0.8
    __segemnts_to_digits: list[set[int]] = [
        {0, 1, 2, 4, 5, 6},
        {2, 5},
        {0, 2, 3, 4, 6},
        {0, 2, 3, 5, 6},
        {1, 2, 3, 5},
        {0, 1, 3, 5, 6},
        {0, 1, 3, 4, 5, 6},
        {0, 2, 5},
        {0, 1, 2, 3, 4, 5, 6},
        {0, 1, 2, 3, 5, 6},
    ]

    def __init__(self, configuration: OcrConfiguration) -> None:
        self._config = configuration

    def read_numer(self, image: np.ndarray) -> int:
        image = self._preprocess_image(image)
        digits_img = [
            SevenSegmentsOcr._move_left(digit) for digit in self._split_digits(image)
        ]
        digits = [
            SevenSegmentsOcr._segemnts_to_digit(self._get_segments(digit_img))
            for digit_img in digits_img
        ]
        number = 0
        for digit in digits:
            number = number * 10 + digit
        return number

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim > 2 else image
        image = cv2.threshold(image, self._config.threshold, 255, cv2.THRESH_BINARY)[1]
        image = cv2.dilate(image, kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        image = cv2.erode(image, kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        return image

    def _split_digits(self, image: np.ndarray) -> list[np.ndarray]:
        width = image.shape[1]
        max_digits = self._config.max_digits
        return [
            image[:, i * width // max_digits : (i + 1) * width // max_digits]
            for i in range(max_digits)
        ]

    def _get_segments(self, image: np.ndarray) -> set[int]:
        threshold = SevenSegmentsOcr.__segment_threshold
        height, width = image.shape
        segemnts = set()
        if self._config.segemnts_definitions:
            for segment_index, frame in self._config.segemnts_definitions.items():
                slice_of_image = image[
                    int(height * frame.top) : int(height * frame.bottom),
                    int(width * frame.left) : int(width * frame.right),
                ]
                covered = sum(sum(slice_of_image))
                area = slice_of_image.shape[0] * slice_of_image.shape[1]
                if covered > threshold * area:
                    segemnts.add(segment_index)
        return segemnts

    @staticmethod
    def _segemnts_to_digit(segemnts: set[int]) -> int:
        for i, digit_segemnts in enumerate(SevenSegmentsOcr.__segemnts_to_digits):
            if segemnts == digit_segemnts:
                return i
        return 0

    @staticmethod
    def _move_left(image: np.ndarray) -> np.ndarray:
        width = image.shape[1]
        shift = 0
        for i, val in enumerate(image.sum(axis=0)[::-1]):
            if val > 0:
                shift = i
                break
        result = np.zeros_like(image)
        result[:, shift:] = image[:, : width - shift]
        return result
