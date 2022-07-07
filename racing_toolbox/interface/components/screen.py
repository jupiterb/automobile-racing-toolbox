import win32gui
import sys
from ctypes import windll

import numpy as np
from PIL import ImageGrab

from interface.exceptions import WindowNotFound
from interface.models import ScreenFrame


class ScreenCapturing:
    def __init__(self, process_name: str, window_size: tuple[int, int]) -> None:
        self._process_name: str = process_name
        self._window_size: tuple[int, int] = window_size
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
        return np.array(ImageGrab.grab(box))

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
        left, top, right, bottom = win32gui.GetWindowRect(window)

        return left, top, right - left, bottom - top
