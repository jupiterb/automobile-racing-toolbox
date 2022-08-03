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


@pytest.fixture
def unvalid_event_detector_parameters() -> list[EventDetectionParameters]:
    return [
        EventDetectionParameters(
            feature_name="min_value_greater_than_max_value",
            min_value=10.0,
            max_value=0.0,
            required_repetitions_in_row=3,
            different_values_required=0,
        ),
        EventDetectionParameters(
            feature_name="negative_different_values_required",
            min_value=0.0,
            max_value=10.0,
            required_repetitions_in_row=3,
            different_values_required=-1,
        ),
    ]


@pytest.fixture
def valid_event_detector_parameters() -> list[EventDetectionParameters]:
    return [
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


@pytest.fixture
def end_of_race_screenshots() -> list[np.ndarray]:
    path_to_images = "assets/screenshots/end_of_race"
    return [
        np.array(Image.open(f"{path_to_images}/{name}"))
        for name in listdir(path_to_images)
    ]


@pytest.fixture
def checkpoint_screenshots() -> list[np.ndarray]:
    path_to_images = "assets/screenshots/checkpoint"
    return [
        np.array(Image.open(f"{path_to_images}/{name}"))
        for name in listdir(path_to_images)
    ]


def perform_detection_on_real_images(
    images: list[np.ndarray], detector: EventDetector, monkeypatch
) -> None:
    detector.reset()
    assert not detector.is_final()

    event_detected = False
    interface = TrainingLocalGameInterface(get_game_config())

    for image in images:

        def mock_get_screenshot(*args, **kwargs):
            return image

        monkeypatch.setattr(Screen, "_get_screenshot", mock_get_screenshot)
        if detector.is_final(interface.perform_ocr(on_last_image=False)):
            event_detected = True
            detector.reset()

    assert event_detected
    assert not detector.is_final()


def test_detector_not_accept_unvalid_parameters(
    unvalid_event_detector_parameters,
) -> None:
    for parameters in unvalid_event_detector_parameters:
        with pytest.raises(ValueError):
            _ = EventDetector([parameters])


def test_event_detection(valid_event_detector_parameters) -> None:
    detector = EventDetector(valid_event_detector_parameters)

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


def test_final_state_detection(monkeypatch, end_of_race_screenshots) -> None:
    detector = EventDetector(get_final_state_detection_parameters())
    perform_detection_on_real_images(end_of_race_screenshots, detector, monkeypatch)


def test_checkpoint_detection(monkeypatch, checkpoint_screenshots) -> None:
    detector = EventDetector(get_checkpoint_detection_parameters())
    perform_detection_on_real_images(checkpoint_screenshots, detector, monkeypatch)
