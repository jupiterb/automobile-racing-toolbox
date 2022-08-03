import numpy as np
import cv2

from interface.ocr.abstract import AbstractOcr
from interface.config import OcrConfiguration, NoResultPolicy


class NineSegmentsOcr(AbstractOcr):

    _element_width_threshold = 0.1
    _element_height_threshold = 0.8
    _dst_elements_height = 40

    _on_error_retries = 2
    _on_error_threshold_reduction = 13

    _digits_segments = [
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

    def __init__(self, config: OcrConfiguration) -> None:
        self._config = config
        self._result = 0
        self._tries_limit = (
            NineSegmentsOcr._on_error_retries if config.try_lower_threshold else 1
        )

    def read_number(self, image: np.ndarray) -> int:
        threshold, tries = self._config.threshold, 0
        while tries < self._tries_limit:
            image = self._preprocess_image(image, threshold)
            elements = self._extract_digits(image)
            digits_segments = [self._get_segments(element) for element in elements]

            if not any(digits_segments):
                self._result = self._policy_decision(self._config.no_elements_policy)
                break

            try:
                digits = [
                    NineSegmentsOcr._digits_segments.index(segments)
                    for segments in digits_segments
                ]
                digits.reverse()
                self._result = sum([digit * 10**i for i, digit in enumerate(digits)])
                break

            except ValueError:
                if self._config.debug:
                    print(f"Error: unknown segments: {digits_segments}")
                threshold -= NineSegmentsOcr._on_error_threshold_reduction

            tries += 1

        if tries == self._tries_limit:
            self._result = self._policy_decision(self._config.no_result_policy)
        return self._result

    def _preprocess_image(self, image: np.ndarray, threshold: int) -> np.ndarray:
        # get binsry image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim > 2 else image
        image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
        # rescale image to standard element height
        nonzero_rows = np.nonzero(image)[0]
        if not any(nonzero_rows):
            return image
        raw_elements_height = nonzero_rows.max() - nonzero_rows.min() + 1
        rescale_factor = NineSegmentsOcr._dst_elements_height / raw_elements_height
        height = int(image.shape[0] * rescale_factor)
        width = int(image.shape[1] * rescale_factor)
        dsize = (width, height)
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
        # connect segements
        image = cv2.dilate(image, kernel=np.array([[1, 1], [1, 1]]))
        image = cv2.dilate(image, kernel=np.array([[1], [1], [1], [1]]))
        # make sure there is no connections between two or more digits
        image = cv2.erode(image, kernel=np.array([[1, 0], [0, 1]], np.uint8))
        image = cv2.erode(image, kernel=np.array([[0, 1], [1, 0]], np.uint8))
        return image

    def _extract_digits(self, image: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = [cv2.boundingRect(contour) for contour in contours]
        if not any(rects):
            return []
        # sort from left to right
        rects.sort(key=lambda rect: rect[0])

        max_height = np.max([height for _, _, _, height in rects])

        min_width = NineSegmentsOcr._element_width_threshold * max_height
        min_height = NineSegmentsOcr._element_height_threshold * max_height

        return [
            NineSegmentsOcr._move_bottom(
                image[top : top + height, left : left + width] / np.max(image),
                max_height,
            )
            for left, top, width, height in rects
            if width > min_width and height > min_height
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

    def _policy_decision(self, policy: NoResultPolicy) -> int:
        if policy == NoResultPolicy.RETURN_ZERO:
            return 0
        if policy == NoResultPolicy.RETURN_NEGATIVE:
            return -1
        else:  # return last policy
            return self._result

    @staticmethod
    def _move_bottom(image: np.ndarray, destination_height: int) -> np.ndarray:
        height, width = image.shape
        result = np.zeros(shape=(destination_height, width))
        result[destination_height - height : destination_height, :] = image
        return result
