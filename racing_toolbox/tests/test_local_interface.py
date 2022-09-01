import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from PIL import Image
from time import sleep

from interface import from_config
from interface.models import SteeringAction
from interface.screen import LocalScreen
from interface.controllers import KeyboardController
from interface.capturing import KeyboardCapturing
from conf import get_game_config


interface = from_config(get_game_config(), KeyboardController, KeyboardCapturing)


def test_perform_ocr(monkeypatch) -> None:
    test_cases = {
        "trackmania_1000x800_0": 84.0,
        "trackmania_1000x800_1": 207.0,
        "trackmania_1000x800_2": 212.0,
        "trackmania_1000x800_3": 205.0,
        "trackmania_1000x800_4": 163.0,
        "trackmania_1000x800_5": 339.0,
        "trackmania_1000x800_6": 0.0,
        "trackmania_1000x800_7": 220.0,
        "trackmania_1000x800_8": 193.0,
        "trackmania_1000x800_9": 268.0,
    }
    for screenshot, value in test_cases.items():

        def mock_get_screenshot(*args, **kwargs):
            return np.array(Image.open(f"assets/screenshots/random/{screenshot}.jpeg"))

        monkeypatch.setattr(LocalScreen, "_grab_image", mock_get_screenshot)
        ocr_result = interface.perform_ocr(on_last=False)
        assert ocr_result == {"speed": value}


def test_keyboard_action() -> None:
    interface.reset()
    sleep(0.01)  # we need to wait a bit for keylogger start

    get_actions_values = lambda actions_set: {
        action: 1.0 if action in actions_set else 0.0 for action in SteeringAction
    }

    test_cases = [
        {SteeringAction.RIGHT, SteeringAction.FORWARD},
        {SteeringAction.BREAK: SteeringAction.LEFT},
        {SteeringAction.FORWARD},
        {},
    ]
    for actions in test_cases:
        print(get_actions_values(actions))
        assert get_actions_values(actions) == interface.read_action()

    action = {SteeringAction.RIGHT: 0.0, SteeringAction.BREAK: 1.0}
    interface.apply_action(action)
    assert interface.read_action()[SteeringAction.BREAK] == 1.0
    assert interface.read_action()[SteeringAction.RIGHT] == 0.0
