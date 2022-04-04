from typing import List
from uuid import UUID
from abc import abstractclassmethod, ABC

from models import GameEnviroment, Agent, Training, GameSystemConfiguration


class AbstractRepository(ABC):

    def __init__(self):
        self.games: List[GameEnviroment] = []

        self._trainings: List[Training] = []
        self._agents: List[Agent] = []

        self._load_game_list()
        self._load_system_configurations()
        self._load_trainings()
        self._load_agents()

    def get_trainings(self, game_id: UUID) -> List[Training]:
        return [training for training in self._trainings if training.game_id == game_id]

    def get_agents(self, game_id: UUID) -> List[Agent]:
        return [agent for agent in self._agents if agent.game_id == game_id]

    def add_training(self, game_id: UUID, name:str) -> UUID:
        pass

    def finish_training(self, training_id: UUID, save_agent: bool):
        pass

    def run_training(self, training_id: UUID):
        pass

    def update_training(self, training: Training):
        pass

    def use_agent(self, agent_id: UUID):
        pass

    def update_game_system_configuration(self, game_id: UUID, system_configuration: GameSystemConfiguration):
        pass

    @abstractclassmethod
    def _load_game_list(self):
        """
        Build list of available enviroments with their global configuration
        """
        pass

    @abstractclassmethod
    def _load_system_configurations(self):
        """
        If game enviroment has system configuration defined, add it 
        """
        pass
    
    @abstractclassmethod
    def _load_trainings(self):
        """
        If game enviroment has any training strated, add it 
        """
        pass

    @abstractclassmethod
    def _load_agents(self):
        """
        If game enviroment has any agent, add it 
        """
        pass

