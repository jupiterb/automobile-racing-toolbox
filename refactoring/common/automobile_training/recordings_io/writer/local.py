import multiprocessing as mp
import numpy as np
import os
from pydantic import BaseModel
import tables as tb
from types import TracebackType
from typing import Optional, Type

from automobile_training.recordings_io.writer.abstract import (
    RecordingWriter,
    RecordingExistsException,
)


class _LocalH5FileRecording(BaseModel):
    frames: tb.EArray
    actions: tb.EArray
    frames_batch: np.ndarray
    actions_batch: np.ndarray


class LocalRecordingWriter(RecordingWriter):
    def __init__(
        self, recording_name: str, recordings_path: str, fps: int, batch_size: int = 32
    ) -> None:
        """Only full batches will be stored"""
        super().__init__(recording_name, fps)
        self._recordings_path = recordings_path
        self._batch_size = batch_size
        self._consumer: Optional[mp.Process] = None
        self._queue = mp.Queue()

    def __enter__(self) -> RecordingWriter:
        file_path = f"{self._recordings_path}/{self._recording_name}.h5"
        if os.path.exists(file_path):
            raise RecordingExistsException(f"Recording {file_path} already exists")
        if not os.path.exists(self._recordings_path):
            os.makedirs(self._recordings_path)
        self._consumer = mp.Process(target=self._consuming, args=(file_path))
        self._consumer.start()
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self._queue.put(False)
        self._consumer.join()

    def put(self, frame: np.ndarray, actions: dict[str, float]) -> None:
        actions_values = np.array(list(actions.values()))
        self._queue.put((frame, actions_values))

    def _consuming(self, file_path: str):
        with tb.File(file_path, "w", driver="H5FD_CORE") as recording_file:
            self._recording_updating(recording_file)

    def _recording_updating(self, recording_file: tb.File):
        recording = None
        place_in_batch = 0
        item = self._queue.get()
        while item:
            frame, actions = item
            if not recording:
                recording = self._init_recording(
                    recording_file, frame.shape, actions.shape
                )
            recording.frames_batch[place_in_batch] = frame
            recording.actions_batch[place_in_batch] = actions
            place_in_batch = (place_in_batch + 1) % self._batch_size
            if not place_in_batch:
                recording.frames.append(recording.frames_batch)
                recording.actions.append(recording.actions_batch)
            item = self._queue.get()

    def _init_recording(
        self, recording_file: tb.File, frame_shape: tuple, actions_shape: tuple
    ) -> _LocalH5FileRecording:
        def create(item_shape: tuple, name: str, atom: tb.Atom):
            batch = np.zeros((self._batch_size, *item_shape))
            where = recording_file.root
            array = recording_file.create_earray(
                where, name, atom, (0, *item_shape), name, chunkshape=batch.shape
            )
            return batch, array

        frames_batch, frames_array = create(frame_shape, "frames", tb.Int8Atom())
        actions_batch, actions_array = create(
            actions_shape, "actions", tb.Float16Atom()
        )

        recording_file.create_array(recording_file.root, "fps", np.array([self._fps]))

        return _LocalH5FileRecording(
            frames=frames_array,
            actions=actions_array,
            frames_batch=frames_batch,
            actions_batch=actions_batch,
        )
