import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from stable_baselines3.common.env_checker import check_env
import numpy as np
from PIL import Image

from interface import LocalGameInterface
from interface.components import Screen
from rl import RealTimeEnviroment
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

    width, height = get_game_config().window_size

    env = RealTimeEnviroment(
        LocalGameInterface(get_game_config()),
        FinalStateDetector(
            [
                FinalValueDetectionParameters(
                    feature_name="speed",
                    min_value=2.0,
                    max_value=None,
                    required_repetitions_in_row=5,
                    not_final_value_required=True,
                )
            ]
        ),
        (height, width, 3),
    )
    check_env(env)
