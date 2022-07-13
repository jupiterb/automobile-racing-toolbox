import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from PIL import Image

from interface import LocalGameInterface
from interface.components import Screen
from rl.final_state import FinalStateDetector
from rl.models import FinalFeatureValueDetectionParameters
from conf import get_game_config


def test_final_state_dector() -> None:
    final_features_values = [
        FinalFeatureValueDetectionParameters(
            feature_name="speed",
            final_value=0.0,
            required_repetitions_in_row=3,
            other_value_required=True,
        ),
        FinalFeatureValueDetectionParameters(
            feature_name="last_checkpoint_detected",
            final_value=1.0,
            required_repetitions_in_row=1,
            other_value_required=False,
        ),
    ]
    detector = FinalStateDetector(final_features_values)

    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})

    assert not detector.is_final({"speed": 10.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})
    assert not detector.is_final({"speed": 0.0, "last_checkpoint_detected": 0.0})

    assert detector.is_final({"speed": 0.0, "last_checkpoint_detected": 1.0})
    assert detector.is_final()

    assert not detector.is_final({"speed": 10.0, "last_checkpoint_detected": 1.0})


def test_integration_with_ocr(monkeypatch) -> None:
    final_features_values = [
        FinalFeatureValueDetectionParameters(
            feature_name="speed",
            final_value=0.0,
            required_repetitions_in_row=5,
            other_value_required=True,
        ),
    ]
    detector = FinalStateDetector(final_features_values)

    assert not detector.is_final()

    end_of_race_detected = False
    interface = LocalGameInterface(get_game_config())

    for screenshot_index in range(447, 460):

        def mock_get_screenshot(*args, **kwargs):
            return np.array(
                Image.open(f"assets/screenshots/end_of_race/ss{screenshot_index}.jpeg")
            )

        monkeypatch.setattr(Screen, "_get_screenshot", mock_get_screenshot)
        end_of_race_detected = detector.is_final(
            interface.perform_ocr(on_last_image=False)
        )

    assert end_of_race_detected
