import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from PIL import Image
from time import sleep

from interface import LocalGameInterface
from interface.models import SteeringAction
from interface.components import Screen
from conf import get_game_config


interface = LocalGameInterface(get_game_config())


def test_perform_ocr(monkeypatch) -> None:
    test_cases = {
        "trackmania_1000x800_0": 84.0,
        "trackmania_1000x800_1": 207.0,
        "trackmania_1000x800_2": 212.0,
        "trackmania_1000x800_3": 205.0,
        "trackmania_1000x800_4": 163.0,
        "trackmania_1000x800_5": 339.0,
        "trackmania_1000x800_6": 0.0,
    }
    for screenshot, value in test_cases.items():

        def mock_get_screenshot(*args, **kwargs):
            return np.array(Image.open(f"assets/screenshots/random/{screenshot}.jpeg"))

        monkeypatch.setattr(Screen, "_get_screenshot", mock_get_screenshot)
        ocr_result = interface.perform_ocr()
        assert ocr_result == {"speed": value}


def test_apply_read_action() -> None:
    interface.reset()
    sleep(0.01)  # we need to wait a bit for keylogger start
    test_cases = [
        [SteeringAction.RIGHT, SteeringAction.FORWARD],
        [SteeringAction.BREAK, SteeringAction.LEFT],
        [SteeringAction.FORWARD],
        [],
    ]
    for actions in test_cases:
        interface.apply_action(actions)
        assert set(actions) == set(interface.read_action())