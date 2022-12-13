from abc import ABC, abstractmethod
import logging
import numpy as np
from racing_toolbox.interface.exceptions import WindowNotFound


logger = logging.getLogger(__name__)


class ScreenProvider(ABC):
    def __init__(self, source: str, screen_size: tuple[int, int]) -> None:
        self._source = source
        self._screen_size = screen_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return *self._screen_size, 3

    def grab_image(self) -> np.ndarray:
        try:
            image = self._grab_image()
        except WindowNotFound as e:
            image = self.__get_default_screen()
            logger.warn(str(e))
        return image

    @abstractmethod
    def _grab_image(self) -> np.ndarray:
        pass

    def __get_default_screen(self) -> np.ndarray:
        return np.zeros(shape=(self._screen_size[0], self._screen_size[1], 3))
