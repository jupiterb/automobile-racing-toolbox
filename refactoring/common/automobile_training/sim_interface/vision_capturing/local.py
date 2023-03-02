import numpy as np
import sys
import win32gui

from ctypes import windll
from PIL import ImageGrab

from automobile_training.sim_interface.vision_capturing.abstract import (
    VisionCapturing,
    SimVisionNotFoundException,
)


class LocalSimCapturing(VisionCapturing):
    def __init__(self, height: int, width: int, window_name: str) -> None:
        if not sys.platform in ["Windows", "win32", "cygwin"]:
            raise NotImplementedError("Works only on Windows platform")
        super().__init__(height, width)
        self.__window_name = window_name
        user32 = windll.user32
        user32.SetProcessDPIAware()

    def get_vision(self) -> np.ndarray:
        return np.array(ImageGrab.grab(self._get_window_rect()))

    def _get_window_rect(self) -> tuple[int, int, int, int]:
        window = win32gui.FindWindow(None, self.__window_name)
        if window == 0:
            raise SimVisionNotFoundException(f"Window {self.__window_name} not found")

        left, top, _, _ = win32gui.GetWindowRect(window)
        win32gui.MoveWindow(window, left, top, self._size[0], self._size[1], True)

        return win32gui.GetWindowRect(window)
