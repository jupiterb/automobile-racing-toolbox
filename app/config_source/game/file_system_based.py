import os
import json

from app.config_source.game.abstract import AbstractGameConfigSource
from racing_toolbox.interface.config import GameConfiguration


class FileSystemGameConfigSource(AbstractGameConfigSource):

    def __init__(self, folder_path: str) -> None:
        self._folder_path = folder_path

    def get_configs(self) -> dict[str, GameConfiguration]:
        if not os.path.exists(self._folder_path):
            raise ValueError(f"{self._folder_path} not found!")
        configs = {}
        for file in os.listdir(self._folder_path):
            file_path = os.path.join(self._folder_path, file)
            if os.path.isfile(file_path):
                with open(file_path) as gp:
                    try:
                        config = GameConfiguration(**json.load(gp))
                        configs[config.full_name] = config
                    except:
                        pass
        return configs

    def add_config(self, config: GameConfiguration):
        pass
