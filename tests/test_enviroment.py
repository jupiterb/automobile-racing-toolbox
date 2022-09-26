import pytest
from stable_baselines3.common.env_checker import check_env
import numpy as np
import gym
from PIL import Image

from racing_toolbox.rl import RealTimeEnviroment
from racing_toolbox.rl.final_state import FinalStateDetector
from racing_toolbox.rl.config import FinalValueDetectionParameters
from racing_toolbox.conf import get_game_config
from racing_toolbox.interface import from_config, GameInterface
from racing_toolbox.interface.screen import LocalScreen
from racing_toolbox.interface.controllers import KeyboardController
from racing_toolbox.rl.wrappers.action import (
    DiscreteActionToVectorWrapper,
    SplitBySignActionWrapper,
)


@pytest.fixture
def my_interface(monkeypatch) -> GameInterface:
    # take screeshot with speed = 0 and same shape like in configuration
    def mock_get_screenshot(*args, **kwargs):
        return np.array(
            Image.open(f"assets/screenshots/random/trackmania_1000x800_0.jpeg")
        )

    monkeypatch.setattr(LocalScreen, "_grab_image", mock_get_screenshot)

    config = get_game_config()
    config.reset_keys_sequence = []
    config.reset_seconds = 0

    interface = from_config(config, KeyboardController)
    return interface


@pytest.fixture
def my_env(my_interface) -> gym.Env:
    detector = FinalStateDetector(
        [
            FinalValueDetectionParameters(
                feature_name="speed",
                min_value=2.0,
                max_value=None,
                required_repetitions_in_row=5,
                not_final_value_required=True,
            )
        ]
    )

    env = RealTimeEnviroment(my_interface, detector)
    return env


def test_gym_implementation(my_env) -> None:
    check_env(my_env)


def test_env_for_gamepad(my_env) -> None:
    env = SplitBySignActionWrapper(my_env, 0)
    check_env(env)


def test_env_for_keyboard(my_env, my_interface):
    available_actions = [
        {"FORWARD"},
        {"FORWARD", "LEFT"},
        {"FORWARD", "RIGHT"},
        {"LEFT"},
        {"RIGHT"},
        set(),
    ]
    env = DiscreteActionToVectorWrapper(
        my_env, available_actions, my_interface.get_possible_actions()
    )
    check_env(env)