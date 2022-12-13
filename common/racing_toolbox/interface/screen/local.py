import win32gui
import sys
from ctypes import windll

import numpy as np
from PIL import ImageGrab

from racing_toolbox.interface.screen.abstract import ScreenProvider
from racing_toolbox.interface.exceptions import WindowNotFound


class LocalScreen(ScreenProvider):
    def __init__(self, source: str, screen_size: tuple[int, int]) -> None:
        super().__init__(source, screen_size)
        user32 = windll.user32
        user32.SetProcessDPIAware()

    def _grab_image(self) -> np.ndarray:
        return np.array(ImageGrab.grab(self._get_window_rect()))

    def _get_window_rect(self) -> tuple[int, int, int, int]:
        if not sys.platform in ["Windows", "win32", "cygwin"]:
            raise NotImplementedError

        window = win32gui.FindWindow(None, self._source)
        if window == 0:
            raise WindowNotFound(self._source)
        left, top, _, _ = win32gui.GetWindowRect(window)
        win32gui.MoveWindow(
            window, left, top, self._screen_size[0], self._screen_size[1], True
        )
        return win32gui.GetWindowRect(window)
