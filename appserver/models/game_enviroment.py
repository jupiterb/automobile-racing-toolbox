from uuid import UUID
from pydantic import BaseModel
from typing import List


from models.game_global_configuration import GameGlobalConfiguration
from models.game_system_configuration import GameSystemConfiguration
from models.training import Training
from models.agent import Agent


class GameEnviroment(BaseModel):
    id: UUID
    name: str
    agents: List[Agent]
    trainings: List[Training]
    global_configuration: GameGlobalConfiguration
    system_configuration: GameSystemConfiguration
