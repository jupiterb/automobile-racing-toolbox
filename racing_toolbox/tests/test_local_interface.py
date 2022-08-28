import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from PIL import Image
from time import sleep

from interface import GameInterfaceBuilder
from interface.models import SteeringAction
from interface.screen import LocalScreen
from conf import get_game_config


interface_builder = GameInterfaceBuilder()
interface_builder.new_interface(get_game_config())
interface_builder.with_keyboard_capturing()
interface_builder.with_keyborad_controller()
interface = interface_builder.build()


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
    test_cases = [
        {SteeringAction.RIGHT: 1.0, SteeringAction.FORWARD: 1.0},
        {SteeringAction.BREAK: 1.0, SteeringAction.LEFT: 1.0},
        {SteeringAction.FORWARD: 1.0},
        {},
    ]
    for actions in test_cases:
        interface.apply_action(actions)
        assert actions == {
            action: value
            for action, value in interface.read_action().items()
            if value > 0
        }

    action = {SteeringAction.RIGHT: 0.0, SteeringAction.BREAK: 1.0}
    interface.apply_action(action)
    assert interface.read_action()[SteeringAction.BREAK] == 1.0
    assert interface.read_action()[SteeringAction.RIGHT] == 0.0
