from stable_baselines3.common.env_checker import check_env
import numpy as np
from PIL import Image

from racing_toolbox.interface import TrainingLocalGameInterface
from racing_toolbox.interface.components import Screen
from racing_toolbox.rl import RealTimeEnviroment
from racing_toolbox.rl.final_state import FinalStateDetector
from racing_toolbox.rl.config import FinalValueDetectionParameters
from racing_toolbox.conf import get_game_config


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
    check_env(env)
