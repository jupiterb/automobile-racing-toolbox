from stable_baselines3.common.env_checker import check_env
from racing_toolbox.environment.wrappers.action import (
    DiscreteActionToVectorWrapper,
    SplitBySignActionWrapper,
)
from tests.test_env.conftest import my_env


def test_gym_implementation(my_env) -> None:
    check_env(my_env)


def test_env_for_gamepad(my_env) -> None:
    env = SplitBySignActionWrapper(my_env, 0)
    check_env(env)


def test_env_for_keyboard(my_env):
    available_actions = {
        "FORWARD": {0, 1, 2},
        "BREAK": set(),
        "RIGHT": {1, 3},
        "LEFT": {2, 4},
    }
    env = DiscreteActionToVectorWrapper(my_env, available_actions)
    check_env(env)
