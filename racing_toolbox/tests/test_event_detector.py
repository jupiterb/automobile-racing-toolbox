import sys
from os import path, listdir

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image

from interface import TrainingLocalGameInterface
from interface.components import Screen
from rl.event import EventDetector
from rl.config import EventDetectionParameters
from conf import (
    get_game_config,
    get_final_state_detection_parameters,
    get_checkpoint_detection_parameters,
)


def test_detector_not_accept_unvalid_parameters() -> None:
    parameters = EventDetectionParameters(
        feature_name="test",
        min_value=10.0,
        max_value=0.0,
        required_repetitions_in_row=3,
        different_values_required=0,
    )
    with pytest.raises(ValueError):
        detector = EventDetector([parameters])
    parameters.max_value = 20
    parameters.different_values_required = -1
    with pytest.raises(ValueError):
        detector = EventDetector([parameters])


def test_event_detection() -> None:
    parameters = [
        EventDetectionParameters(
            feature_name="speed",
            min_value=float("-inf"),
            max_value=2.0,
            required_repetitions_in_row=3,
            different_values_required=2,
        ),
        EventDetectionParameters(
            feature_name="last_checkpoint_detected",
            min_value=1.0,
            max_value=float("inf"),
            required_repetitions_in_row=1,
            different_values_required=0,
        ),
    ]
    detector = EventDetector(parameters)

    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})

    assert not detector.is_final({"speed": 5.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 5.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert detector.is_final({"speed": 0.0, "last_checkpoint_detected": 1.0})
    assert detector.is_final()

    assert not detector.is_final({"speed": 5.0, "last_checkpoint_detected": 1.0})
    assert not detector.is_final()

    detector.reset()
    assert not detector.is_final()


def test_final_state_detection(monkeypatch) -> None:
    detector = EventDetector(get_final_state_detection_parameters())

    assert not detector.is_final()

    end_of_race_detected = False
    interface = TrainingLocalGameInterface(get_game_config())

    path_to_images = "assets/screenshots/end_of_race"
    for image_name in listdir(path_to_images):

        def mock_get_screenshot(*args, **kwargs):
            return np.array(Image.open(f"{path_to_images}/{image_name}"))

        monkeypatch.setattr(Screen, "_get_screenshot", mock_get_screenshot)
        end_of_race_detected = detector.is_final(
            interface.perform_ocr(on_last_image=False)
        )

    assert end_of_race_detected


def test_checkpoint_detection(monkeypatch) -> None:
    detector = EventDetector(get_checkpoint_detection_parameters())

    assert not detector.is_final()

    checpoint_detected = False
    interface = TrainingLocalGameInterface(get_game_config())

    path_to_images = "assets/screenshots/checkpoint"
    for image_name in listdir(path_to_images):

        def mock_get_screenshot(*args, **kwargs):
            return np.array(Image.open(f"{path_to_images}/{image_name}"))

        monkeypatch.setattr(Screen, "_get_screenshot", mock_get_screenshot)
        if detector.is_final(interface.perform_ocr(on_last_image=False)):
            checpoint_detected = True
            detector.reset()

    assert checpoint_detected
    assert not detector.is_final()
