import pytest
import time

from automobile_training.sim_interface.action_capturing import (
    ActionCapturing,
    KeyboardCapturing,
)
from automobile_training.sim_interface.action_controller import (
    ActionController,
    KeyboardController,
)


def action_to_key() -> dict[str, str]:
    return {"forward": "up", "break": "down", "left": "left", "right": "right"}


def key_to_action() -> dict[str, str]:
    return {k: a for a, k in action_to_key().items()}


@pytest.mark.parametrize(
    "capturing,controller",
    [(KeyboardCapturing(key_to_action()), KeyboardController(action_to_key(), []))],
)
def test_controller_actions_are_captured(
    capturing: ActionCapturing, controller: ActionController
):
    capturing.start()
    time.sleep(0.1)

    actions = {i: 0.0 for i in controller.possible_actions}
    controller.apply(actions)
    assert capturing.get_actions() == actions

    actions[list(controller.possible_actions)[0]] = 1.0
    controller.apply(actions)
    assert capturing.get_actions() == actions

    capturing.stop()
    assert set(capturing.get_actions()) == controller.possible_actions
