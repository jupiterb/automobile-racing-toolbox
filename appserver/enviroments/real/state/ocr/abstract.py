from abc import abstractmethod
import numpy as np
import cv2

from schemas import GameGlobalConfiguration
from schemas.game.feature_extraction import MorphologyOperationType


class AbstractOcr:
    def __init__(self, global_configuration: GameGlobalConfiguration) -> None:
        self._ocr_velocity_params = global_configuration.ocr_velocity_params

    @abstractmethod
    def read_number(self, image: np.ndarray) -> int:
        pass

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        dilated_eroded = binary
        for operation in self._ocr_velocity_params.morphology_operations_combination:
            if operation.type == MorphologyOperationType.DILATING:
                dilated_eroded = cv2.dilate(
                    dilated_eroded,
                    kernel=np.array(operation.kernel, np.uint8),
                    iterations=operation.iterations,
                )
            elif operation.type == MorphologyOperationType.EROSION:
                dilated_eroded = cv2.erode(
                    dilated_eroded,
                    kernel=np.array(operation.kernel, np.uint8),
                    iterations=operation.iterations,
                )

        return dilated_eroded

    def _separated_digits(self, image: np.ndarray) -> list[np.ndarray]:
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        rects = [cv2.boundingRect(contour) for contour in contours]
        rects = [
            (left, top, width, height)
            for left, top, width, height in rects
            if width >= self._ocr_velocity_params.min_width
            and height >= self._ocr_velocity_params.min_height
        ]
        rects.sort(key=lambda rect: rect[0])
        return [
            self.__normalize_shape(image[top : top + height, left : left + width])
            for left, top, width, height in rects
        ]

    def __normalize_shape(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] < self._ocr_velocity_params.shape_width:
            image = cv2.copyMakeBorder(
                image,
                0,
                0,
                self._ocr_velocity_params.shape_width - image.shape[1],
                0,
                cv2.BORDER_CONSTANT,
                value=0,
            )
        return cv2.resize(
            image,
            (
                self._ocr_velocity_params.shape_width,
                self._ocr_velocity_params.shape_height,
            ),
            interpolation=cv2.INTER_AREA,
        )
