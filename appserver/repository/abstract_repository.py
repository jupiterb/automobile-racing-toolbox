from abc import ABC, abstractclassmethod
from typing import List
from schemas import GameEnviromentBase, TrainingBase, Training, TrainingParameters


class AbstractRepository(ABC):

    @abstractclassmethod
    def view_games(self) -> List[GameEnviromentBase]:
        pass

    @abstractclassmethod
    def view_trainings(self, game_name: str) -> List[TrainingBase]:
        pass

    @abstractclassmethod
    def create_trainings(self, game_name: str, training_base: TrainingBase) -> Training:
        pass

    @abstractclassmethod
    def delete_training(self, game_name: str, training_name):
        pass

    @abstractclassmethod
    def run_training(self, game_name: str, training_name):
        pass

    @abstractclassmethod
    def stop_training(self, game_name: str, training_name):
        pass

    @abstractclassmethod
    def update_training(game_name: str, training_name: str, training_parameters: TrainingParameters) -> Training:
        pass
