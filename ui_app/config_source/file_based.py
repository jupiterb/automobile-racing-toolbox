import os
import json
from typing import Generic

from ui_app.config_source.abstract import (
    AbstractConfigSource,
    RacingToolboxConfiguration,
)


class _SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)


class FileSysteConfigSource(AbstractConfigSource, Generic[RacingToolboxConfiguration]):
    def __init__(self, folder_path: str) -> None:
        self._folder_path = folder_path

    def get_configs(self) -> dict[str, RacingToolboxConfiguration]:
        config_cls = self.__orig_class__.__args__[0]
        if not os.path.exists(self._folder_path):
            raise ValueError(f"{self._folder_path} not found!")
        configs = {}
        for file in os.listdir(self._folder_path):
            file_path = os.path.join(self._folder_path, file)
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    try:
                        config = config_cls(**json.load(f))
                        name = file.split(".")[0]
                        configs[name] = config
                    except:
                        pass
        return configs

    def add_config(self, name: str, config: RacingToolboxConfiguration):
        if not os.path.exists(self._folder_path):
            raise ValueError(f"{self._folder_path} not found!")
        new_config_path = f"{self._folder_path}/{name}.json"
        with open(new_config_path, "w") as f:
            json.dump(config.dict(), f, cls=_SetEncoder)
