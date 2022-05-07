from abc import ABC, abstractmethod

from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State


class RealTimeWrapper(ABC):

    def __init__(self,
        global_configuration: GameGlobalConfiguration, 
        system_configuration: GameSystemConfiguration
    ) -> None:
        self._global_configuration = global_configuration
        self._system_configuration = system_configuration

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def step(self, action: Action) -> State:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def read_state(self) -> State:
        pass

    @abstractmethod
    def apply_action(self, action: Action):
        pass

    @abstractmethod
    def read_action(self) -> Action:
        pass
