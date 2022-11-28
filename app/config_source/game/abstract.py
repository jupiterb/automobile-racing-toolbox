from abc import abstractmethod
from racing_toolbox.interface.config import GameConfiguration


class AbstractGameConfigSource:
    @abstractmethod
    def get_configs(self) -> dict[str, GameConfiguration]:
        pass

    @abstractmethod
    def add_config(self, config: GameConfiguration):
        pass
