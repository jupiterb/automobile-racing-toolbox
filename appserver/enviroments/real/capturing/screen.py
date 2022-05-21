import pygetwindow
import numpy as np
from typing import Optional
from PIL import ImageGrab

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
        window = self._maximize_window()
        if self._specified_window_rect:
            left, top, width, height = self._specified_window_rect
        else:
            left, top, width, height = (
                window.left,
                window.top,
                window.width,
                window.height,
            )
        box = (
            int(left + width * screen_frame.left),
            int(top + height * screen_frame.top),
            int(left + width * screen_frame.right),
            int(top + height * screen_frame.bottom),
        )
        image = np.array(ImageGrab.grab(box))
        return image

    def _maximize_window(self) -> pygetwindow.Window:
        same_name_windows = pygetwindow.getWindowsWithTitle(self._process_name)
        if not same_name_windows:
            raise WindowNotFound(self._process_name)
        window = same_name_windows[0]
        if not window.isMaximized:
            window.maximize()
        return window
