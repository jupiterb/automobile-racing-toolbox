from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
import numpy as np


class RecordingWriter(AbstractContextManager, ABC):
    def __init__(self, recording_name: str, fps: int) -> None:
        self._recording_name = recording_name
        self._fps = fps

    @abstractmethod
    def put(self, frame: np.ndarray, actions: dict[str, float]) -> None:
        """Puts to recording context frame and actions"""
        pass
