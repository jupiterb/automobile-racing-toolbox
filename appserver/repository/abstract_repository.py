from copy import copy
from typing import List, Tuple, Dict
from uuid import UUID, uuid4
from abc import abstractclassmethod, ABC
from fastapi.encoders import jsonable_encoder

from models import GameEnviroment, Agent, Training, GameSystemConfiguration


class AbstractRepository(ABC):

    def __init__(self):
        self.games: List[GameEnviroment] = []

        self._trainings: Dict[Training] = []
        self._agents: Dict[Agent] = []

        self._load_game_list()
        self._load_system_configurations()
        self._load_trainings()
        self._load_agents()

    def get_trainings(self, game_name: str) -> List[Training]:
        game_id = self._get_game_id_from(game_name)
        return [training for training in self._trainings.values() if training.game_id == game_id]

    def get_agents(self, game_name: str) -> List[Agent]:
        game_id = self._get_game_id_from(game_name)
        return [agent for agent in self._agents.values() if agent.game_id == game_id]

    def add_training(self, game_name: str, name :str, full_name :str) -> Tuple[bool, Training]:
        game_id = self._get_game_id_from(game_name)
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
        game_id = self._get_game_id_from(game_name)
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

    def add_agent(self, game_name: str, agent_name: str, agent: Agent) -> Tuple[bool, Training]:
        game_id = self._get_game_id_from(game_name)
        same_name_agents = [agent for agent in self._agents.values() 
            if agent.endpoint_name == agent_name and agent.game_id == game_id]
        if any(same_name_agents):
            return (False, same_name_agents[0])
        else:
            self._trainings[agent.id] = agent
            return (True, agent)

    def use_agent(self, game_name: str, agent_name: str):
        # TODO: run agent
        pass

    def remove_agent(self, game_name: str, agent_name: str):
        game_id = self._get_game_id_from(game_name)
        self._trainings = [agent for agent in self._agents.values() 
            if agent.game_id != game_id and agent.endpoint_name != agent_name]

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

    def _get_game_id_from(self, game_name: str) -> UUID:
        return [game.id for game in self.games if game.endpoint_name == game_name][0]

    def _get_training_id_from(self, game_name: str, training_name: str):
        game_id = self._get_game_id_from(game_name)
        return [training.id for training in self._trainings.values() 
            if training.game_id == game_id and training.endpoint_name == training_name][0]
