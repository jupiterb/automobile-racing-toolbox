from abc import ABC
from repository.abstract_repository import AbstractRepository
from typing import List, Tuple
from schemas import GameEnviromentBase, TrainingBase, Training, TrainingParameters


class InMemoryRepository(AbstractRepository, ABC):

    def __init__(self):
        pass

    def view_games(self) -> List[GameEnviromentBase]:
        pass

    def view_trainings(self, game_name: str) -> List[TrainingBase]:
        pass

    def create_training(self, game_name: str, training_base: TrainingBase) -> Tuple[bool, Training]:
        pass

    def delete_training(self, game_name: str, training_name):
        pass

    def run_training(self, game_name: str, training_name) -> bool:
        pass

    def stop_training(self, game_name: str, training_name) -> bool:
        pass

    def update_training(self, game_name: str, training_name: str, training_parameters: TrainingParameters) -> Training:
        pass
