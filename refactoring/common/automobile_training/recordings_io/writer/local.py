import numpy as np
from types import TracebackType
from typing import Optional, Type

from automobile_training.recordings_io.writer.abstract import RecordingWriter


class LocalRecordingWriter(RecordingWriter):
    def __init__(
        self, recording_name: str, recordings_path: str, fps: int, batch_size: int = 32
    ) -> None:
        super().__init__(recording_name, fps)
        self._recordings_path = recordings_path
        self._batch_size = batch_size

    def __enter__(self) -> RecordingWriter:
        return super().__enter__()

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return super().__exit__(__exc_type, __exc_value, __traceback)

    def put(self, frame: np.ndarray, actions: dict[str, float]) -> None:
        return super().put(frame, actions)
