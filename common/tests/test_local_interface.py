import pytest
import numpy as np
from PIL import Image
from time import sleep

from racing_toolbox.interface import from_config
from racing_toolbox.interface.screen import LocalScreen
from racing_toolbox.interface.controllers import KeyboardController
from racing_toolbox.interface.capturing import KeyboardCapturing
from racing_toolbox.observation.utils.ocr import OcrTool, SevenSegmentsOcr
from tests.conftest import game_conf
from tests import TEST_DIR


@pytest.fixture
def interface(game_conf):
    return from_config(game_conf, KeyboardController, KeyboardCapturing)


def test_integration_with_ocr(monkeypatch, game_conf, interface) -> None:
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

    ocr_tool = OcrTool(game_conf.ocrs, SevenSegmentsOcr)

    for screenshot, value in test_cases.items():

        def mock_get_screenshot(*args, **kwargs):
            return np.array(
                Image.open(TEST_DIR / f"assets/screenshots/random/{screenshot}.jpeg")
            )

        monkeypatch.setattr(LocalScreen, "_grab_image", mock_get_screenshot)
        image = interface.grab_image()
        ocr_result = ocr_tool.perform(image)
        assert ocr_result == {"speed": value}


def test_keyboard_action(interface) -> None:
    interface.reset()
    sleep(0.01)  # we need to wait a bit for keylogger start

    actions = interface.get_possible_actions()

    get_actions_values = lambda actions_set: {
        action: 1.0 if action in actions_set else 0.0 for action in actions
    }

    test_cases = [
        {actions[0], actions[1]},
        {actions[2]},
        {actions[3], actions[0]},
        {},
    ]
    for test_case in test_cases:
        interface.apply_action(get_actions_values(test_case))
        assert get_actions_values(test_case) == interface.read_action()

    action = {actions[0]: 0.0, actions[1]: 1.0}
    interface.apply_action(action)
    assert interface.read_action()[actions[1]] == 1.0
    assert interface.read_action()[actions[0]] == 0.0