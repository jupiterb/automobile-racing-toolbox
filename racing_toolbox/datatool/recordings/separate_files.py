import os

import numpy as np
from PIL import Image
import pandas as pd

from racing_toolbox.datatool.recordings import RecorderDataService
from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.models import Recording


class SeparateFilesRecordingsService(RecorderDataService):

    _path_to_data = "/data"

    def __init__(self) -> None:
        self._data_frame: pd.DataFrame | None = None
        self._fullpath: str = ""
        self._sequence_number: int = 0

    def start_streaming(
        self, game_name: str, user_name: str, recording_name: str, fps: int
    ) -> None:
        self._fullpath = f"{os.path.dirname(__file__)}/{SeparateFilesRecordingsService._path_to_data}/{game_name}/{user_name}/{recording_name}"
        if os.path.exists(self._fullpath):
            raise ItemExists(game_name, user_name, recording_name)
        os.makedirs(self._fullpath)
        self._data_frame = pd.DataFrame()
        self._sequence_number = 0

    def stop_streaming(self) -> None:
        if self._data_frame is not None:
            self._data_frame.to_csv(f"{self._fullpath}/data.csv")

    def put_observation(
        self,
        image: np.ndarray,
        numerical_data: dict[str, float],
        actions_values: dict[str, float],
    ) -> None:
        if self._data_frame is None:
            return
        image_name = self._save_as_jpeg(image)
        datarow = pd.DataFrame(
            {"id": [self._sequence_number], "screenshot": [image_name]}
        )
        for name, value in numerical_data.items():
            datarow[name] = value
        for action, value in actions_values.items():
            datarow[action] = value
        self._data_frame = pd.concat(
            [self._data_frame, datarow], ignore_index=True, axis=0
        )
        self._sequence_number += 1

    def get_recording(
        self, game_name: str, user_name: str, recording_name
    ) -> Recording:
        raise NotImplementedError()

    def _save_as_jpeg(self, image) -> str:
        name_with_extension = f"ss{self._sequence_number}.jpeg"
        Image.fromarray(image).save(f"{self._fullpath}/{name_with_extension}")
        return name_with_extension
