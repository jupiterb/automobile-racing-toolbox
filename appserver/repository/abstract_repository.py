from typing import List
from models import GameEnviroment

class AbstractRepository(object):

    def __init__(self):
        self._games = List[GameEnviroment]
        self._load_game_list()

    def _load_game_list(self):
        pass