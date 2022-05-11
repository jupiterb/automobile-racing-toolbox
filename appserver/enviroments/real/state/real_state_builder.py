import numpy as np
import cv2
from PIL import Image
import random
import time

from schemas import State, GameGlobalConfiguration


class RealStateBuilder():

    def __init__(self, global_configuration: GameGlobalConfiguration) -> None:
        self._state = State()
        self._ocr_velocity_params = global_configuration.ocr_velocity_params

    def reset(self):
        self._state = State()

    def add_features_from_screenshot(self, screenshot: np.ndarray):
        self._state.screenshot_numpy_array = screenshot

    def add_velocity_with_ocr(self, screenshot: np.ndarray):
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, self._ocr_velocity_params.binarization_threshold, 255, cv2.THRESH_BINARY)[1]
        
        dilated_eroded = binary
        for iterations, kernel in self._ocr_velocity_params.dilate_erode_combination:
            if iterations > 0:
                dilated_eroded = cv2.dilate(dilated_eroded, kernel=np.array(kernel, np.uint8), iterations=iterations)
            elif iterations < 0:
                dilated_eroded = cv2.erode(dilated_eroded, kernel=np.array(kernel, np.uint8), iterations=-iterations)

        transformed_screenshot = dilated_eroded

        digits_captured = []
        contours = cv2.findContours(transformed_screenshot, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if width >= self._ocr_velocity_params.min_width and height >= self._ocr_velocity_params.min_height:
                digits_captured.append(transformed_screenshot[left:left+width, top:top:height])

    def get_result(self) -> State:
        return self._state
