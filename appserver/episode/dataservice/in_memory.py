import os

import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional

from episode.dataservice.abstract import AbstractEpisodesRecordingsDataService
from schemas import Episode, EpisodeRecording, Action


class InMemoryEpisodesRecordingsDataService(AbstractEpisodesRecordingsDataService):

    _path_to_data = "/data"

    def __init__(self) -> None:
        super().__init__()

    def save(self, game_id: str, episode: Episode):
        fullpath = InMemoryEpisodesRecordingsDataService._get_fullpath(
            game_id, episode.id
        )
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

        def save_as_jpeg(image, name) -> str:
            name_with_extension = f"{name}.jpeg"
            Image.fromarray(image).save(f"{fullpath}/{name_with_extension}")
            return name_with_extension

        def keys_to_str(keys: Optional[set[str]]) -> str:
            result = "+"
            if keys:
                for key in keys:
                    result += f"{key}+"
            return result

        if episode.recording:
            data = [
                [save_as_jpeg(image, f"{i}_scr"), velocity, keys_to_str(action.keys)]
                for i, (image, velocity, action) in enumerate(
                    episode.recording.recording
                )
            ]
            data_frame = pd.DataFrame(
                data=data, columns=["screenshot_file", "velocity", "keys"]
            )
            data_frame.to_csv(
                InMemoryEpisodesRecordingsDataService._get_csv_path(game_id, episode.id)
            )

    def get_episode(self, game_id: str, episode_id: str) -> Episode:
        data_frame = pd.read_csv(
            InMemoryEpisodesRecordingsDataService._get_csv_path(game_id, episode_id)
        )
        episode = Episode(id=episode_id, recording=EpisodeRecording())
        fullpath = InMemoryEpisodesRecordingsDataService._get_fullpath(
            game_id, episode_id
        )
        assert episode.recording
        for _, row in data_frame.iterrows():
            screenshot_file = row["screenshot_file"]
            with Image.open(f"{fullpath}/{screenshot_file}") as screenshot:
                velocity: int = row["velocity"]
                keys: set[str] = {key for key in row["keys"].split("+") if len(key) > 0}
                episode.recording.recording.append(
                    (np.asarray(screenshot), velocity, Action(keys=keys))
                )
        return episode

    @staticmethod
    def _get_fullpath(game_id: str, episode_id: str) -> str:
        return f"{os.path.dirname(__file__)}/{InMemoryEpisodesRecordingsDataService._path_to_data}/{game_id}/{episode_id}"

    @staticmethod
    def _get_csv_path(game_id: str, episode_id: str) -> str:
        return f"{InMemoryEpisodesRecordingsDataService._get_fullpath(game_id, episode_id)}/data.csv"
