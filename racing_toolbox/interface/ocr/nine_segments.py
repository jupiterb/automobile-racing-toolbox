import numpy as np
import cv2

from interface.ocr.abstract import AbstractOcr
from interface.models import OcrConfiguration, NoResultPolicy


class NineSegmentsOcr(AbstractOcr):

    _element_size_threshold: float = 0.2
    _dst_elements_height = 40
    _min_elements_height = 12
    _on_error_retries = 2
    _on_error_threshold_change = 13
    _digits_segments: list[set[int]] = [
        {0, 1, 3, 5, 7, 8},
        {2, 6},
        {0, 3, 4, 5, 8},
        {0, 3, 4, 7, 8},
        {1, 3, 4, 7},
        {0, 1, 4, 7, 8},
        {0, 1, 4, 5, 7, 8},
        {0, 3, 7},
        {0, 1, 3, 4, 5, 7, 8},
        {0, 1, 3, 4, 7, 8},
    ]

    def __init__(self, configuration: OcrConfiguration) -> None:
        self._config = configuration
        self._last_number = 0
        self._retries_left = NineSegmentsOcr._on_error_retries

    def read_number(self, image: np.ndarray) -> int:
        image = self._preprocess_image(image)
        elements = self._split_digits(image)
        digits_segments = [self._get_segments(element) for element in elements]

        if not any(digits_segments):
            return self._policy_decision(self._config.lack_of_elements_policy, image)

        try:
            digits = [
                NineSegmentsOcr._digits_segments.index(segments)
                for segments in digits_segments
            ]
            digits.reverse()
            self._last_number = sum([digit * 10**i for i, digit in enumerate(digits)])
        except ValueError:
            print(f"Error: unknown segments: {digits_segments}")
            return self._policy_decision(self._config.unknown_elements_policy, image)

        self._retries_left = NineSegmentsOcr._on_error_retries
        return self._last_number

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim > 2 else image
        image = cv2.threshold(image, self._config.threshold, 255, cv2.THRESH_BINARY)[1]
        # rescale image to standard element height
        nonzero_rows = np.nonzero(image)[0]
        if not any(nonzero_rows):
            return image
        raw_elements_height = nonzero_rows.max() - nonzero_rows.min() + 1
        if raw_elements_height < NineSegmentsOcr._min_elements_height:
            return np.zeros_like(image)
        rescale_factor = NineSegmentsOcr._dst_elements_height / raw_elements_height
        height = int(image.shape[0] * rescale_factor)
        width = int(image.shape[1] * rescale_factor)
        dsize = (width, height)
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
        # connect segements
        image = cv2.dilate(image, kernel=np.array([[1, 1], [1, 1]]))
        # additional, vertical connection
        image = cv2.dilate(image, kernel=np.array([[1], [1], [1], [1]]))
        # make sure there is no connections between two or more digits
        image = cv2.erode(image, kernel=np.array([[1, 0], [0, 1]], np.uint8))
        image = cv2.erode(image, kernel=np.array([[0, 1], [1, 0]], np.uint8))
        return image

    def _split_digits(self, image: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = [cv2.boundingRect(contour) for contour in contours]
        if not any(rects):
            return []
        # sort from left to right
        rects.sort(key=lambda rect: rect[0])

        elements = [
            image[top : top + height, left : left + width] / np.max(image)
            for left, top, width, height in rects
        ]
        max_height = np.max([height for _, _, _, height in rects])
        min_size = NineSegmentsOcr._element_size_threshold * max_height

        return [
            NineSegmentsOcr._move_bottom(element, max_height)
            for element in elements
            if np.max(element.shape) > min_size and element.shape[0] > element.shape[1]
        ]

    def _get_segments(self, image: np.ndarray) -> set[int]:
        if not self._config.segments_coordinates:
            return set()
        height, width = image.shape
        segments = {
            segment_index
            for segment_index, (x, y) in self._config.segments_coordinates.items()
            if image[int(height * x), int(width * y)] > 0.0
        }
        # only '1' has 2 and 6 segemnt, but many times other segemnts are also detected
        fixed_one = NineSegmentsOcr._digits_segments[1]
        return segments if not fixed_one.issubset(segments) else fixed_one

    def _policy_decision(self, policy: NoResultPolicy, image: np.ndarray) -> int:
        if self._config.try_lower_threshold and self._retries_left > 0:
            self._retries_left -= 1
            original_threshold = self._config.threshold
            self._config.threshold -= NineSegmentsOcr._on_error_threshold_change
            number = self.read_number(image)
            self._config.threshold = original_threshold
            return number
        else:
            self._retries_left = NineSegmentsOcr._on_error_retries
        if policy == NoResultPolicy.RETURN_ZERO:
            return 0
        if policy == NoResultPolicy.RETURN_NEGATIVE:
            return -1
        else:  # return last policy
            return self._last_number

    @staticmethod
    def _move_bottom(image: np.ndarray, destination_height: int) -> np.ndarray:
        height, width = image.shape
        result = np.zeros(shape=(destination_height, width))
        result[destination_height - height : destination_height, :] = image
        return result
