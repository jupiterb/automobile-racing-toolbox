from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

from automobile_training.recordings_io.model import RecordingModel


class RecordingNotFoundException(Exception):
    def __init__(self, description: str) -> None:
        super().__init__(description)


class RecordingReader(ABC):
    def __init__(self) -> None:
        super().__init__()

    @contextmanager
    @abstractmethod
    def get(self) -> Generator[RecordingModel, None, None]:
        """Reads recording in context"""
        pass
