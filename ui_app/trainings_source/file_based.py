import os
import pickle

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import ModelConfig

from ui_app.trainings_source.abstract import AbstractModelsSource, WeightsAndConfigs
from ui_app.config_source.file_based import FileSysteConfigSource as ConfigSource


class FileSysteModelsSource(AbstractModelsSource):
    def __init__(self, folder_path: str) -> None:
        self._folder_path = folder_path

    def get_models(self) -> dict[str, WeightsAndConfigs]:
        if not os.path.exists(self._folder_path):
            raise ValueError(f"{self._folder_path} not found!")
        models = {}
        for name in os.listdir(self._folder_path):
            model_path = os.path.join(self._folder_path, name)
            if os.path.isdir(model_path):
                try:
                    (
                        game_config,
                        env_config,
                        model_config,
                    ) = FileSysteModelsSource._get_configs(model_path)
                    weights = FileSysteModelsSource._get_weights(model_path)
                    models[name] = (weights, model_config, game_config, env_config)
                except Exception as e:
                    pass
        return models

    @staticmethod
    def _get_weights(model_path: str):
        checkpoint_path = f"{model_path}/checkpoint"
        try:
            with open(checkpoint_path, "rb") as f:
                model = pickle.load(f)
            value = model["worker"]
            return pickle.loads(value)["state"]["default_policy"]["weights"]
        except PermissionError:
            return {}

    @staticmethod
    def _get_configs(model_path: str):
        return (
            list(ConfigSource[GameConfiguration](model_path).get_configs().values())[0],
            list(ConfigSource[EnvConfig](model_path).get_configs().values())[0],
            list(ConfigSource[ModelConfig](model_path).get_configs().values())[0],
        )
