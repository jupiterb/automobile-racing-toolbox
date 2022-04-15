from abc import ABC, abstractmethod
from typing import List, Tuple

from schemas import Game, Training, TrainingParameters, \
                    GameGlobalConfiguration, GameSystemConfiguration


class AbstractRepository(ABC):

    @abstractmethod
    def view_games(self) -> List[Game]:
        pass

    @abstractmethod
    def view_trainings(self, game_id: str) -> List[Training]:
        pass

    @abstractmethod
    def view_training(self, game_id: str, training_id: str) -> Training:
        pass

    @abstractmethod
    def create_training(self, game_id: str, training_id: str, description: str) -> Tuple[bool, Training]:
        pass

    @abstractmethod
    def delete_training(self, game_id: str, training_id: str):
        pass

    @abstractmethod
    def get_full_training_configuration(self, game_id: str, training_id: str) -> \
        Tuple[GameSystemConfiguration, GameGlobalConfiguration, TrainingParameters]:
        pass
