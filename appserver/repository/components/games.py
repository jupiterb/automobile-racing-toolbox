from typing import Dict, List
from schemas import GameEnviroment, GameSystemConfiguration
from uuid import UUID


class Games(object):

    def __init__(self):
        self._games: Dict[str, GameEnviroment] = {}

    def get_view(self) -> List[GameEnviroment]:
        return self._games.values()

    def get_game_id_from(self, game_name: str) -> UUID:
        return [game.id for game in self._games if game.endpoint_name == game_name][0]

    def update_game_system_configuration(self, game_name: str, system_configuration: GameSystemConfiguration):
        pass
