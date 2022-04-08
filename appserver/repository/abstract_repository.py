from abc import ABC, abstractclassmethod
from typing import List, Tuple

from schemas import GameEnviromentBase, TrainingBase, Training, TrainingParameters, \
                    GameGlobalConfiguration, GameSystemConfiguration
from training import TrainingManager


class AbstractRepository(ABC):

    def __init__(self):
        self.training_manager = TrainingManager()

    @abstractclassmethod
    def view_games(self) -> List[GameEnviromentBase]:
        pass

    @abstractclassmethod
    def view_trainings(self, game_name: str) -> List[TrainingBase]:
        pass

    @abstractclassmethod
    def view_training(self, game_name: str, training_name: str) -> Training:
        pass

    @abstractclassmethod
    def create_training(self, game_name: str, training_base: TrainingBase) -> Tuple[bool, Training]:
        pass

    @abstractclassmethod
    def delete_training(self, game_name: str, training_name: str):
        pass

    @abstractclassmethod
    def update_training(self, game_name: str, training_name: str, training_parameters: TrainingParameters) -> Training:
        pass

    @abstractclassmethod
    def _get_full_training_configuration(self, game_name: str, training_name: str) -> \
        Tuple[GameSystemConfiguration, GameGlobalConfiguration, TrainingParameters]:
        pass

    def run_training(self, game_name: str, training_name: str):
        system_configuration, global_configuration, training_parameters = self._get_full_training_configuration(game_name, training_name)
        self.training_manager.run_training(system_configuration, global_configuration, training_parameters)

    def stop_training(self, game_name: str, training_name: str):
        self.training_manager.stop_training()
