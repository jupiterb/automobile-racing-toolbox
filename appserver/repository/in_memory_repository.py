from abc import ABC
from uuid import UUID, uuid4
from typing import List, Tuple

from schemas import GameEnviromentBase, TrainingBase, Training, TrainingParameters, \
                    GameEnviroment, GameGlobalConfiguration, GameSystemConfiguration
from repository.abstract_repository import AbstractRepository


class InMemoryRepository(AbstractRepository, ABC):

    def __init__(self):
        game_id = uuid4()
        self._games: List[GameEnviroment] = [
            GameEnviroment(
                id = game_id,
                global_configuration = GameGlobalConfiguration(),
                system_configuration = GameSystemConfiguration(),
                endpoint_name = "trackmania",
                full_name = "Trackmania Nations Forever",
                trainings = [
                    Training(
                        id = uuid4(),
                        game_id = game_id,
                        endpoint_name = "first",
                        full_name = "First Training",
                        parameters = TrainingParameters()
                    )
                ]
            )
        ]

    def view_games(self) -> List[GameEnviromentBase]:
        return [GameEnviromentBase(endpoint_name = game.endpoint_name, full_name = game.full_name) 
                for game in self._games]

    def view_trainings(self, game_name: str) -> List[TrainingBase]:
        trainings = self._from_name(game_name).trainings
        return [TrainingBase(endpoint_name = training.endpoint_name, full_name = training.full_name)
                for training in trainings]

    def view_training(self, game_name: str, training_name: str) -> Training:
        return [training for training in self._from_name(game_name).trainings 
                if training.endpoint_name == training_name]

    def create_training(self, game_name: str, training_base: TrainingBase) -> Tuple[bool, Training]:
        game = self._from_name(game_name)
        same_name_trainings = [training for training in game.trainings 
                                if training.endpoint_name == training_base.endpoint_name]

        if any(same_name_trainings):
            return (False, same_name_trainings[0])

        else:
            new_training = Training(
                id = uuid4(),
                game_id = game.id,
                endpoint_name = training_base.endpoint_name,
                full_name = training_base.full_name,
                parameters = TrainingParameters()
            )
            game.trainings.append(new_training)
            return (True, new_training)

    def delete_training(self, game_name: str, training_name: str):
        game = self._from_name(game_name)
        new_trainings = [training for training in game.trainings if training.endpoint_name != training_name]
        game.trainings = new_trainings

    def run_training(self, game_name: str, training_name: str):
        pass

    def stop_training(self, game_name: str, training_name: str):
        pass

    def update_training(self, game_name: str, training_name: str, training_parameters: TrainingParameters) -> Training:
        game = self._from_name(game_name)
        training = [training for training in game.trainings if training.endpoint_name == training_name][0]
        training.parameters = training_parameters
        return training

    def _from_name(self, game_name: str) -> GameEnviroment:
        return [game for game in self._games if game.endpoint_name == game_name][0]
