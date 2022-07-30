from interface import TrainingLocalGameInterface
from interface.config import GameConfiguration, SteeringAction
from interface.components import KeyboardCapturing


class FullLocalGameInterface(TrainingLocalGameInterface):
    def __init__(self, configuration: GameConfiguration) -> None:
        super().__init__(configuration)
        self._keyboard_capturing = KeyboardCapturing(self._available_keys)
        self._keys_mapping = {
            key: action
            for action, key in self._configuration.discrete_actions_mapping.items()
        }

    def reset(self) -> None:
        super().reset()
        self._keyboard_capturing.reset()

    def read_action(self) -> list[SteeringAction]:
        return [
            self._keys_mapping[key] for key in self._keyboard_capturing.get_pressed()
        ]
