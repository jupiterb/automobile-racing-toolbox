from abc import ABC, abstractmethod
import numpy as np

from interface.models import ScreenFrame


class ScreenProvider(ABC):

    def __init__(self, source: str, screen_size: tuple[int, int], default_frame: ScreenFrame) -> None:
        self._source = source
        self._screen_size = screen_size
        self.__last_image: np.ndarray = np.zeros(
            shape=(screen_size[0], screen_size[1], 3)
        )
        self.__default_frame = default_frame

    def grab_image(
        self, screen_frame: ScreenFrame | None = None, on_last: bool = False
    ) -> np.ndarray:
        frame = screen_frame if screen_frame else self.__default_frame
        image = self.__last_image if on_last else self._grab_image()
        self.__last_image = image
        height, width, _ = image.shape
        return image[
            int(height * frame.top) : int(height * frame.bottom),
            int(width * frame.left) : int(width * frame.right),
        ]

    @abstractmethod
    def _grab_image(self) -> np.ndarray:
        pass
    