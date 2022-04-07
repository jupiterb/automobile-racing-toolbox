from uuid import UUID
from pydantic import BaseModel

from models.game_global_configuration import GameGlobalConfiguration
from models.game_system_configuration import GameSystemConfiguration


class GameEnviroment(BaseModel):
    id: UUID
    full_name: str = "Trackmania Nations Forever"
    endpoint_name: str = "trackmania"
    global_configuration: GameGlobalConfiguration
    system_configuration: GameSystemConfiguration
