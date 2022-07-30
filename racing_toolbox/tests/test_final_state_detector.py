import sys
from os import path, listdir

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image

from interface import TrainingLocalGameInterface
from interface.components import Screen
from rl.event import EventsDetector
from rl.config import EventDetectionParameters
from conf import get_game_config


def test_detector_not_accept_unvalid_values_ranges() -> None:
    final_features_values = [
        EventDetectionParameters(
            feature_name="test",
            min_value=10.0,
            max_value=0.0,
            required_repetitions_in_row=3,
            not_event_values_required=True,
        ),
    ]
    with pytest.raises(ValueError):
        detector = EventsDetector(final_features_values)


def test_final_state_detection() -> None:
    final_values_detection_parameters = [
        EventDetectionParameters(
            feature_name="speed",
            min_value=2.0,
            max_value=None,
            required_repetitions_in_row=3,
            not_event_values_required=True,
        ),
        EventDetectionParameters(
            feature_name="last_checkpoint_detected",
            min_value=None,
            max_value=0.0,
            required_repetitions_in_row=1,
            not_event_values_required=False,
        ),
    ]
    detector = EventsDetector(final_values_detection_parameters)

    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})

    assert not detector.is_final({"speed": 10.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})

    assert detector.is_final({"speed": 5.0, "last_checkpoint_detected": 1.0})
    assert detector.is_final()

    assert not detector.is_final({"speed": 2.0, "last_checkpoint_detected": 0.0})

    assert detector.is_final({"speed": 5.0, "last_checkpoint_detected": 1.0})
    detector.reset()
    assert not detector.is_final()


def test_integration_with_ocr(monkeypatch) -> None:
    final_features_values = [
        EventDetectionParameters(
            feature_name="speed",
            min_value=2.0,
            max_value=None,
            required_repetitions_in_row=5,
            not_event_values_required=True,
        ),
    ]
    detector = EventsDetector(final_features_values)

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
