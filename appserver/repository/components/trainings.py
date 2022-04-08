from typing import List, Tuple, Dict
from uuid import uuid4
from fastapi.encoders import jsonable_encoder

from schemas import Agent, Training
from repository.components import Games


class Trainings(object):

    def __init__(self, games: Games):
        self._trainings: Dict[Training] = {}
        self._games: Games = games

    def get_view(self, game_name: str) -> List[Training]:
        game_id = self._get_game_id_from(game_name)
        return [training for training in self._trainings.values() if training.game_id == game_id]

    def add_training(self, game_name: str, name :str, full_name :str) -> Tuple[bool, Training]:
        game_id = self._games.get_game_id_from(game_name)
        same_name_trainings = [training for training in self._trainings.values() 
            if training.endpoint_name == name and training.game_id == game_id]
        if any(same_name_trainings):
            return (False, same_name_trainings[0])
        else:
            new_training = Training(
                id = uuid4(),
                game_id = game_id,
                endpoint_name = name,
                full_name = full_name
            )
            self._trainings[new_training.id] = new_training
            return (True, new_training)

    def remove_training(self, game_name: str, training_name: str):
        # TODO: finish training if running
        game_id = self._games.get_game_id_from(game_name)
        self._trainings = [training for training in self._trainings.values() 
            if training.game_id != game_id and training.endpoint_name != training_name]

    def run_training(self, game_name: str, training_name: str):
        # TODO: run training
        pass

    def finish_training(self, game_name: str, training_name: str):
        # TODO: finish training
        pass

    def update_training(self, game_name: str, training_name: str, training_data: Training):
        training_id = self._get_training_id_from(game_name, training_name)
        stored_training = self._trainings[training_id]
        stored_training_model = Training(**stored_training)
        update_training = training_data.dict(exclude_unset=True)
        update_training = stored_training_model.copy(update=update_training)
        self._trainings[training_id] = jsonable_encoder(update_training)
        # TODO: apply updated parameters

    def get_agent_from_training(self, game_name: str, training_name: str) -> Agent:
        # TODO: extract agent from training
        pass

    def _get_training_id_from(self, game_name: str, training_name: str):
        game_id = self._games.get_game_id_from(game_name)
        return [training.id for training in self._trainings.values() 
            if training.game_id == game_id and training.endpoint_name == training_name][0]
