from abc import ABC
from typing import List, Tuple, Dict
from fastapi import HTTPException, status

from schemas import Training, TrainingParameters, \
                    Game, GameGlobalConfiguration, GameSystemConfiguration
from repository.abstract_repository import AbstractRepository


class InMemoryRepository(AbstractRepository, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.games: Dict[str, Game] = {}

    def view_games(self) -> List[Game]:
        return [game for game in self.games.values()]

    def view_trainings(self, game_id: str) -> List[Training]:
        return self._try_get_game(game_id).trainings

    def add_game(self, game_id: str, description: str) -> Tuple[bool, Game]:
        if game_id in self.games:
            return (False, self.games[game_id])
        else:
            new_game = Game(
                id = game_id,
                description = description
            )
            self.games[game_id] = new_game
            return (True, new_game)

    def delete_game(self, game_id: str):
        if game_id in self.games:
            del self.games[game_id]

    def setup_game(self, game_id: str, system_configuration: GameSystemConfiguration) -> Game:
        game = self._try_get_game(game_id)
        game.system_configuration = system_configuration
        return game

    def view_training(self, game_id: str, training_id: str) -> Training:
        return self._try_get_game_training(game_id, training_id)[1]

    def create_training(self, game_id: str, training_id: str, description: str) -> Tuple[bool, Training]:
        game = self._try_get_game(game_id)
        same_id_trainings = [training for training in game.trainings if training.id == training_id]
        if any(same_id_trainings):
            return (False, same_id_trainings[0])
        else:
            new_training = Training(
                id = training_id,
                game_id = game_id,
                description = description
            )
            game.trainings.append(new_training)
            return (True, new_training)

    def delete_training(self, game_id: str, training_id: str):
        if game_id in self.games:
            trainings = self.games[game_id].trainings
            self.games[game_id].trainings = [training for training in trainings if training.id != training_id]

    def setup_training(self, game_id: str, training_id: str, training_parameters: TrainingParameters) -> Training:
        _, training = self._try_get_game_training(game_id, training_id)
        training.parameters = training_parameters
        return training

    def get_full_training_configuration(self, game_id: str, training_id: str) -> \
        Tuple[GameSystemConfiguration, GameGlobalConfiguration, TrainingParameters]:
        game, training = self._try_get_game_training(game_id, training_id)
        return (game.system_configuration, game.global_configuration, training.parameters)

    def _try_get_game(self, game_id: str) -> Game:
        if game_id in self.games:
            return self.games[game_id]
        else:
            raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = f"Game {game_id} not found")

    def _try_get_game_training(self, game_id: str, training_id: str) -> Tuple[Game, Training]:
        game = self._try_get_game(game_id)
        trainings = [training for training in game.trainings if training.id == training_id]
        if any(trainings):
            return (game, trainings[0])
        else:
            raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = f"Training {training_id} not found")
