import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from stable_baselines3.common.env_checker import check_env
import numpy as np
from PIL import Image

from interface import TrainingLocalGameInterface
from interface.components import Screen
from rl import RealTimeEnvironment
from rl.event import EventDetector
from rl.config import EventDetectionParameters
from conf import (
    get_game_config,
    get_checkpoint_detection_parameters,
    get_final_state_detection_parameters,
)


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
    final_state_detector = EventDetector(get_final_state_detection_parameters())
    checkpoint_detector = EventDetector(get_checkpoint_detection_parameters())

    env = RealTimeEnvironment(interface, final_state_detector, checkpoint_detector)
    check_env(env)
