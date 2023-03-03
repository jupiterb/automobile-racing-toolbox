import pytest
import time

from automobile_training.sim_interface.input_capturing import (
    InputCapturing,
    KeyboardCapturing,
)
from automobile_training.sim_interface.input_controller import (
    InputController,
    KeyboardController,
)


def input_to_key() -> dict[str, str]:
    return {"forward": "up", "break": "down", "left": "left", "right": "right"}


def key_to_input() -> dict[str, str]:
    return {k: i for i, k in input_to_key().items()}


@pytest.mark.parametrize(
    "inputs_capturing,controller",
    [(KeyboardCapturing(key_to_input()), KeyboardController(input_to_key(), []))],
)
def test_controller_inputs_are_captured(
    inputs_capturing: InputCapturing, controller: InputController
):
    inputs_capturing.start()
    time.sleep(0.1)

    inputs = {i: 0.0 for i in controller.possible_inputs}
    controller.apply(inputs)
    assert inputs_capturing.get_inputs() == inputs

    inputs[list(controller.possible_inputs)[0]] = 1.0
    controller.apply(inputs)
    assert inputs_capturing.get_inputs() == inputs

    inputs_capturing.stop()
    assert set(inputs_capturing.get_inputs()) == controller.possible_inputs
