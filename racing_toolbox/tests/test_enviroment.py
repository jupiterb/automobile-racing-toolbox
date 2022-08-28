import sys
from os import path
import pytest

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from stable_baselines3.common.env_checker import check_env
import numpy as np
import gym
from PIL import Image

from interface import GameInterfaceBuilder
from interface.screen import LocalScreen
from interface.models import SteeringAction

from rl import RealTimeEnviroment
from rl.final_state import FinalStateDetector
from rl.config import FinalValueDetectionParameters
from rl.wrappers import (
    DiscreteActionToVectorWrapper,
    ZeroThresholdingActionWrapper,
    StandardActionRangeToPositiveWarapper,
)

from conf import get_game_config


@pytest.fixture
def my_env(monkeypatch) -> gym.Env:
    # take screeshot with speed = 0 and same shape like in configuration
    def mock_get_screenshot(*args, **kwargs):
        return np.array(
            Image.open(f"assets/screenshots/random/trackmania_1000x800_0.jpeg")
        )

    monkeypatch.setattr(LocalScreen, "_grab_image", mock_get_screenshot)

    config = get_game_config()
    config.reset_keys_sequence = []
    config.reset_seconds = 0

    interface_builder = GameInterfaceBuilder()
    interface_builder.new_interface(config)
    interface_builder.with_keyborad_controller()
    interface = interface_builder.build()

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

    env = RealTimeEnviroment(interface, detector)
    return env


def test_gym_implementation(my_env) -> None:
    check_env(my_env)


def test_env_for_gamepad(my_env) -> None:
    env = ZeroThresholdingActionWrapper(
        my_env, [SteeringAction.BREAK, SteeringAction.FORWARD]
    )
    check_env(env)


def test_env_for_keyboard(my_env):
    available_actions = [
        {SteeringAction.FORWARD},
        {SteeringAction.FORWARD, SteeringAction.LEFT},
        {SteeringAction.FORWARD, SteeringAction.RIGHT},
        {SteeringAction.LEFT},
        {SteeringAction.RIGHT},
        set(),
    ]
    env = DiscreteActionToVectorWrapper(my_env, available_actions)
    check_env(env)
