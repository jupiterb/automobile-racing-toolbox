from contextlib import contextmanager
import os
import tables as tb
from typing import Generator


from automobile_training.recordings_io.model import RecordingModel
from automobile_training.recordings_io.reader.abstract import (
    RecordingReader,
    RecordingNotFoundException,
)


class LocalRecordingReader(RecordingReader):
    def __init__(self, path: str) -> None:
        self._path = path

    @contextmanager
    def get(self) -> Generator[RecordingModel, None, None]:
        if not os.path.exists(self._path):
            raise RecordingNotFoundException(
                f"Recording not found under path {self._path}"
            )
        with tb.File(self._path, driver="H5FD_CORE") as file:
            yield RecordingModel(
                fps=int(file.root.fps[0]),
                frames=file.root.frames,
                actions=file.root.actions,
            )
