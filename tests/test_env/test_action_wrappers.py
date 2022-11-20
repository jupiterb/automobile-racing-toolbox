import py
import pytest
import numpy as np
from racing_toolbox.environment.wrappers.action import DiscreteActionToVectorWrapper
from tests.test_env.conftest import my_env


def test_discrete_actions_wrapper(my_env) -> None:
    actions_number = 6
    available_actions = {
        "FORWARD": {0, 1, 2},
        "BREAK": set(),
        "RIGHT": {1, 3},
        "LEFT": {2, 4},
    }
    env = DiscreteActionToVectorWrapper(my_env, available_actions)
    for i in range(actions_number):
        action = env.action(i)
        assert i == env.reverse_action(action)
        for present, expected in zip(action, available_actions.values()):
            assert present and i in expected or not present and i not in expected

    with pytest.raises(ValueError):
        env.action(actions_number)

    with pytest.raises(ValueError):
        env.reverse_action(np.array([1.0, 1.0, 1.0, 1.0]))
