from abc import ABC, abstractmethod
import numpy as np

from interface.models import ScreenFrame
from interface.exceptions import WindowNotFound


class ScreenProvider(ABC):
    def __init__(
        self, source: str, screen_size: tuple[int, int], default_frame: ScreenFrame
    ) -> None:
        self._source = source
        self._screen_size = screen_size
        self.__last_image = self.__get_default_screen()
        self.__default_frame = default_frame

    def grab_image(
        self, screen_frame: ScreenFrame | None = None, on_last: bool = False
    ) -> np.ndarray:
        frame = screen_frame if screen_frame else self.__default_frame
        try:
            image = self.__last_image if on_last else self._grab_image()
        except WindowNotFound as e:
            image = self.__get_default_screen()
            print(str(e))
        self.__last_image = image
        height, width, _ = image.shape
        return image[
            int(height * frame.top) : int(height * frame.bottom),
            int(width * frame.left) : int(width * frame.right),
        ]

    @abstractmethod
    def _grab_image(self) -> np.ndarray:
        pass

    def __get_default_screen(self) -> np.ndarray:
        return np.zeros(shape=(self._screen_size[0], self._screen_size[1], 3))
