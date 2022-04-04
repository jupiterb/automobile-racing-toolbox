from unicodedata import name
from models import GameEnviroment, GameGlobalConfiguration, GameSystemConfiguration, Agent, Training
from repository.abstract_repository import AbstractRepository
from uuid import uuid4
from abc import ABC


class InMemoryRepository(AbstractRepository, ABC):

    def __init__(self):
        super().__init__()

    def _load_game_list(self):
        self.games = [
            GameEnviroment(
                id = uuid4(),
                name = "Trackmania Nations Forever",
                global_configuration = GameGlobalConfiguration(),
                system_configuration = GameSystemConfiguration()
            ),
            GameEnviroment(
                id = uuid4(),
                name = "Forza Horizon 5",
                global_configuration = GameGlobalConfiguration(),
                system_configuration = GameSystemConfiguration()
            ),
        ]

    def _load_system_configurations(self):
        pass
    
    def _load_trainings(self):
        for game in self.games:
            self._trainings.append(Training(
                id = uuid4(),
                game_id = game.id,
                name = game.name + ": first traning"
            ))
            self._trainings.append(Training(
                id = uuid4(),
                game_id = game.id,
                name = game.name + ": second traning"
            ))

    def _load_agents(self):
        for game in self.games:
            self._agents.append(Agent(
                id = uuid4(),
                game_id = game.id,
                name = game.name + ": first agent"
            ))
            self._agents.append(Agent(
                id = uuid4(),
                game_id = game.id,
                name = game.name + ": second agent"
            )) 
