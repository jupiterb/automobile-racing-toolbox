from abc import abstractmethod
from typing import Generic, TypeVar

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig, ModelConfig


RacingToolboxConfiguration = TypeVar(
    "RacingToolboxConfiguration",
    GameConfiguration,
    EnvConfig,
    TrainingConfig,
    ModelConfig,
)


class AbstractConfigSource(Generic[RacingToolboxConfiguration]):
    @abstractmethod
    def get_configs(self) -> dict[str, RacingToolboxConfiguration]:
        pass

    @abstractmethod
    def add_config(self, config: RacingToolboxConfiguration):
        pass
