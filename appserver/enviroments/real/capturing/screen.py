import win32gui
from ctypes import windll
import numpy as np
from typing import Optional
from PIL import ImageGrab
import sys
import cv2

from schemas import ScreenFrame
from utils.custom_exceptions import WindowNotFound


class ScreenCapturing:
    def __init__(
        self,
        process_name: str,
        apply_grayscale: bool,
        specified_window_rect: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        self._process_name: str = process_name
        self._specified_window_rect: Optional[
            tuple[int, int, int, int]
        ] = specified_window_rect
        self._apply_grayscale: bool = apply_grayscale

        user32 = windll.user32
        user32.SetProcessDPIAware()

    def grab_image(self, screen_frame: ScreenFrame) -> np.ndarray:
        left, top, width, height = self._get_window_rect()
        box = (
            int(left + width * screen_frame.left),
            int(top + height * screen_frame.top),
            int(left + width * screen_frame.right),
            int(top + height * screen_frame.bottom),
        )
        image = np.array(ImageGrab.grab(box))
        return (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if self._apply_grayscale else image
        )

    def _get_window_rect(self) -> tuple[int, int, int, int]:
        if self._specified_window_rect:
            return self._specified_window_rect

        if not sys.platform in ["Windows", "win32", "cygwin"]:
            raise NotImplementedError

        window = win32gui.FindWindow(None, self._process_name)
        if window == 0:
            raise WindowNotFound(self._process_name)
        left, top, right, bottom = win32gui.GetWindowRect(window)

        return left, top, right - left, bottom - top
