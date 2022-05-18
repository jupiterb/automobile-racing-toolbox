from abc import ABC, abstractmethod
from pynput.keyboard import Key
from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State


class RealGameInterface(ABC):
    def __init__(
        self,
        global_configuration: GameGlobalConfiguration,
        system_configuration: GameSystemConfiguration,
    ) -> None:
        self._global_configuration = global_configuration
        self._system_configuration = system_configuration

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def read_frame(self) -> State:
        pass

    @abstractmethod
    def apply_keyboard_action(self, action: list[Key]):
        pass

    @abstractmethod
    def read_action(self) -> Action:
        pass
