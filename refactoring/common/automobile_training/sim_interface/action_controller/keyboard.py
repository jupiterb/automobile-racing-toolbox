from pynput.keyboard import Controller, Key

from automobile_training.sim_interface.action_controller.abstract import (
    ActionController,
)


class KeyboardController(ActionController):
    def __init__(
        self, action_to_key_mapping: dict[str, str], reset_sequence: list[str]
    ) -> None:
        self._controller = Controller()
        self._action_to_key_mapping = action_to_key_mapping
        self._reset_sequence = reset_sequence

    @property
    def possible_actions(self) -> set[str]:
        return set(self._action_to_key_mapping)

    def reset(self) -> None:
        for key in self._action_to_key_mapping.values():
            key = self.get_key(key)
            self._controller.release(key)
        for key in self._reset_sequence:
            key = self.get_key(key)
            self._controller.press(key)
            self._controller.release(key)

    def apply(self, actions: dict[str, float]) -> None:
        actual_actions = {
            action
            for action, value in actions.items()
            if action in self._action_to_key_mapping and value > 0
        }
        for action in actual_actions:
            key = self.get_key(self._action_to_key_mapping[action])
            self._controller.press(key)
        for action in set(self._action_to_key_mapping) - actual_actions:
            key = self.get_key(self._action_to_key_mapping[action])
            self._controller.release(key)

    @staticmethod
    def get_key(key_name: str) -> Key:
        return Key[key_name]
