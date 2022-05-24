import pygetwindow
import numpy as np
from typing import Any, Optional
from PIL import ImageGrab
import sys

from schemas import ScreenFrame
from utils.custom_exceptions import WindowNotFound


class ScreenCapturing:
    def __init__(
        self,
        process_name: str,
        specified_window_rect: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        self._process_name: str = process_name
        self._specified_window_rect: Optional[
            tuple[int, int, int, int]
        ] = specified_window_rect

    def grab_image(self, screen_frame: ScreenFrame) -> np.ndarray:
        left, top, width, height = self._get_window_rect()
        box = (
            int(left + width * screen_frame.left),
            int(top + height * screen_frame.top),
            int(left + width * screen_frame.right),
            int(top + height * screen_frame.bottom),
        )
        image = np.array(ImageGrab.grab(box))
        return image

    def _get_window_rect(self) -> tuple[int, int, int, int]:
        if self._specified_window_rect:
            return self._specified_window_rect

        if not sys.platform in ["Windows", "win32", "cygwin"]:
            raise NotImplementedError

        window = self.__windows_activate_window()
        return window.left, window.top, window.width, window.height

    def __windows_activate_window(self) -> pygetwindow.Window:
        same_name_windows = pygetwindow.getWindowsWithTitle(self._process_name)
        if not same_name_windows:
            raise WindowNotFound(self._process_name)
        window = same_name_windows[0]
        return window
