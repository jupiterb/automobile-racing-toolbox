from pynput.keyboard import Controller, Key

from automobile_training.sim_interface.input_controller.abstract import InputController


class KeyboardController(InputController):
    def __init__(
        self, input_to_key_mapping: dict[str, str], reset_sequence: list[str]
    ) -> None:
        self._controller = Controller()
        self._input_to_key_mapping = input_to_key_mapping
        self._reset_sequence = reset_sequence

    @property
    def possible_inputs(self) -> set[str]:
        return set(self._input_to_key_mapping)

    def reset(self) -> None:
        for key in self._input_to_key_mapping.values():
            key = self.get_key(key)
            self._controller.release(key)
        for key in self._reset_sequence:
            key = self.get_key(key)
            self._controller.press(key)
            self._controller.release(key)

    def apply(self, inputs: dict[str, float]) -> None:
        inputs_set = {
            input
            for input, value in inputs.items()
            if input in self._input_to_key_mapping and value > 0
        }
        for input in inputs_set:
            key = self.get_key(self._input_to_key_mapping[input])
            self._controller.press(key)
        for input in set(self._input_to_key_mapping) - inputs_set:
            key = self.get_key(self._input_to_key_mapping[input])
            self._controller.release(key)

    @staticmethod
    def get_key(input: str) -> Key:
        return Key[input]
