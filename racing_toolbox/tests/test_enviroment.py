import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from stable_baselines3.common.env_checker import check_env
import numpy as np
from PIL import Image

from interface import TrainingLocalGameInterface
from interface.components import Screen
from rl import RealTimeEnviroment
from rl.enviroment import _rescale, _to_grayscale
from rl.final_state import FinalStateDetector
from rl.models import FinalValueDetectionParameters
from conf import get_game_config


def test_gym_implementation(monkeypatch) -> None:
    # take screeshot with speed = 0 and same shape like in configuration
    def mock_get_screenshot(*args, **kwargs):
        return np.array(
            Image.open(f"assets/screenshots/random/trackmania_1000x800_0.jpeg")
        )

    monkeypatch.setattr(Screen, "_get_screenshot", mock_get_screenshot)

    config = get_game_config()
    config.reset_keys_sequence = []
    config.reset_seconds = 0

    interface = TrainingLocalGameInterface(config)
    observation_shape = _rescale(_to_grayscale(interface.grab_image()), 100).shape
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

    env = RealTimeEnviroment(interface, detector, observation_shape)
    check_env(env)
