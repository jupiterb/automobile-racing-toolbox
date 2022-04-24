import sys
import win32gui
import win32con
import numpy as np
from PIL import ImageGrab

from schemas import State, ScreenFrame
from utils.custom_exceptions import WindowNotFound


class ScreenCapturing():

    def __init__(self, process_name: str, driving_screen_frame: ScreenFrame) -> None:
        self._process_name: str = process_name
        self._driving_screen_frame: ScreenFrame = driving_screen_frame

    def capture_state(self) -> State:
        window_frame = self._get_window_frame()
        image = self._grab_image(window_frame)
        return self._get_state_from_image(image)

    def _get_window_frame(self) -> ScreenFrame:
        if not sys.platform in ['Windows', 'win32', 'cygwin']:
            raise NotImplementedError

        hwnd = win32gui.FindWindow(None, self._process_name)
        if hwnd == 0:
            raise WindowNotFound(self._process_name)

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

        (left, top, right, bottom) = win32gui.GetWindowRect(hwnd)
        return ScreenFrame(top=top, bottom=bottom, left=left, right=right)

    def _grab_image(self, window_frame: ScreenFrame) -> np.ndarray:
        x, y = window_frame.right - window_frame.left, window_frame.bottom - window_frame.top
        box = (
            int(window_frame.left + x * self._driving_screen_frame.left), 
            int(window_frame.top + y * self._driving_screen_frame.top), 
            int(window_frame.left + x * self._driving_screen_frame.right), 
            int(window_frame.top + y * self._driving_screen_frame.bottom), 
        )
        image = np.array(ImageGrab.grab(box))
        return image

    def _get_state_from_image(self, image: np.ndarray) -> State:
        return State()
