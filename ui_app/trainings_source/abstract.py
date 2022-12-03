from abc import abstractmethod
from collections import namedtuple


WeightsAndConfigs = namedtuple(
    "WeightsAndConfigs", ["weights", "model_config", "game_config", "env_config"]
)


class AbstractModelsSource:
    @abstractmethod
    def get_models(self) -> dict[str, WeightsAndConfigs]:
        pass
