import sys
import win32gui
import win32con
import numpy as np
from PIL import ImageGrab

from schemas import ScreenFrame
from utils.custom_exceptions import WindowNotFound


class ScreenCapturing():

    def __init__(self, process_name: str) -> None:
        self._process_name: str = process_name

    def grab_image(self, screen_frame: ScreenFrame) -> np.ndarray:
        (left, top, right, bottom) = self._get_window_rect()
        width, height = right - left, bottom - top
        box = (
            int(left + width * screen_frame.left), 
            int(top + height * screen_frame.top), 
            int(left + width * screen_frame.right), 
            int(top + height * screen_frame.bottom), 
        )
        image = np.array(ImageGrab.grab(box))
        return image

    def _get_window_rect(self) -> tuple[int, int, int, int]:
        if not sys.platform in ['Windows', 'win32', 'cygwin']:
            raise NotImplementedError

        hwnd = win32gui.FindWindow(None, self._process_name)
        if hwnd == 0:
            raise WindowNotFound(self._process_name)

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

        return win32gui.GetWindowRect(hwnd)
