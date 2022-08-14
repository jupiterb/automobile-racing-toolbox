import win32gui
import sys
from ctypes import windll

import numpy as np
from PIL import ImageGrab

from interface.exceptions import WindowNotFound
from interface.models import ScreenFrame


class Screen:
    def __init__(self, process_name: str, window_size: tuple[int, int]) -> None:
        self._process_name: str = process_name
        self._window_size: tuple[int, int] = window_size
        self._last_screenshot: np.ndarray = np.zeros(
            shape=(window_size[0], window_size[1], 3)
        )
        user32 = windll.user32
        user32.SetProcessDPIAware()

    def grab_image(
        self, screen_frame: ScreenFrame, on_last: bool = False
    ) -> np.ndarray:
        image = self._get_screenshot(on_last)
        self._last_screenshot = image
        height, width, _ = image.shape
        return image[
            int(height * screen_frame.top) : int(height * screen_frame.bottom),
            int(width * screen_frame.left) : int(width * screen_frame.right),
        ]

    def _get_screenshot(self, on_last: bool) -> np.ndarray:
        return (
            self._last_screenshot
            if on_last
            else np.array(ImageGrab.grab(self._get_window_rect()))
        )

    def _get_window_rect(self) -> tuple[int, int, int, int]:
        if not sys.platform in ["Windows", "win32", "cygwin"]:
            raise NotImplementedError

        window = win32gui.FindWindow(None, self._process_name)
        if window == 0:
            raise WindowNotFound(self._process_name)
        left, top, _, _ = win32gui.GetWindowRect(window)
        win32gui.MoveWindow(
            window, left, top, self._window_size[0], self._window_size[1], True
        )
        return win32gui.GetWindowRect(window)
