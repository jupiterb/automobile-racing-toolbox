from abc import ABC
from typing import List, Tuple

from schemas import Training, TrainingParameters, \
                    Game, GameGlobalConfiguration, GameSystemConfiguration
from repository.abstract_repository import AbstractRepository


class InMemoryRepository(AbstractRepository, ABC):

    