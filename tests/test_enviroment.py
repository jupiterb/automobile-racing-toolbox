import pytest
from stable_baselines3.common.env_checker import check_env
import numpy as np
import gym
from PIL import Image
from racing_toolbox.conf.example_configuration import get_game_config

from racing_toolbox.environment import RealTimeEnviroment
from racing_toolbox.environment.final_state import FinalStateDetector
from racing_toolbox.environment.config import FinalValueDetectionParameters
from racing_toolbox.interface import from_config, GameInterface
from racing_toolbox.interface.screen import LocalScreen
from racing_toolbox.interface.controllers import KeyboardController
from racing_toolbox.environment.wrappers.action import (
    DiscreteActionToVectorWrapper,
    SplitBySignActionWrapper,
)
from racing_toolbox.observation.utils.ocr import OcrTool, SevenSegmentsOcr
from tests import TEST_DIR


@pytest.fixture
def my_interface(monkeypatch, game_conf) -> GameInterface:
    # take screeshot with speed = 0 and same shape like in configuration
    def mock_get_screenshot(*args, **kwargs):
        return np.array(
            Image.open(
                TEST_DIR / f"assets/screenshots/random/trackmania_1000x800_0.jpeg"
            )
        )

    monkeypatch.setattr(LocalScreen, "_grab_image", mock_get_screenshot)

    config = game_conf
    config.reset_keys_sequence = []
    config.reset_seconds = 0

    interface = from_config(config, KeyboardController)
    return interface


@pytest.fixture
def my_env(my_interface) -> gym.Env:
    ocr_tool = OcrTool(get_game_config().ocrs, SevenSegmentsOcr)

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

    env = RealTimeEnviroment(my_interface, ocr_tool, detector)
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
