from abc import ABC, abstractmethod

from schemas import Episode


class AbstractEpisodesRecordingsDataService(ABC):

    @abstractmethod
    def save(self, game_id: str, episode: Episode):
        pass

    @abstractmethod
    def get_episode(self, game_id: str, episode_id: str)  -> Episode:
        pass
