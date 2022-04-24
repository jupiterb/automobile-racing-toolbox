from enviroments.common import RealTimeWrapper

from schemas import GameGlobalConfiguration, GameSystemConfiguration, Action, State


class RealGameWrapper(RealTimeWrapper):

    def __init__(self, 
        global_configuration: GameGlobalConfiguration,
        system_configuration: GameSystemConfiguration
    ) -> None:
        super().__init__(global_configuration, system_configuration)

    def step(self, action: Action) -> State:
        return super().step(action)

    def reset(self) -> State:
        return super().reset()

    def read_state(self) -> State:
        return super().read_state()

    def apply_action(self, action: Action):
        return super().apply_action(action)

    def read_action(self) -> Action:
        return super().read_action()
