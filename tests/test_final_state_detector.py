import pytest
from os import listdir
import numpy as np
from PIL import Image

from racing_toolbox.environment.final_state import FinalStateDetector
from racing_toolbox.environment.config import FinalValueDetectionParameters
from racing_toolbox.conf import get_game_config
from racing_toolbox.interface import from_config
from racing_toolbox.interface.screen import LocalScreen
from racing_toolbox.observation.utils.ocr import OcrTool, SevenSegmentsOcr
from tests import TEST_DIR


def test_detector_not_accept_unvalid_values_ranges() -> None:
    final_features_values = [
        FinalValueDetectionParameters(
            feature_name="test",
            min_value=10.0,
            max_value=0.0,
            required_repetitions_in_row=3,
            not_final_value_required=True,
        ),
    ]
    with pytest.raises(ValueError):
        detector = FinalStateDetector(final_features_values)


def test_final_state_detection() -> None:
    final_values_detection_parameters = [
        FinalValueDetectionParameters(
            feature_name="speed",
            min_value=2.0,
            max_value=None,
            required_repetitions_in_row=3,
            not_final_value_required=True,
        ),
        FinalValueDetectionParameters(
            feature_name="last_checkpoint_detected",
            min_value=None,
            max_value=0.0,
            required_repetitions_in_row=1,
            not_final_value_required=False,
        ),
    ]
    detector = FinalStateDetector(final_values_detection_parameters)

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
        FinalValueDetectionParameters(
            feature_name="speed",
            min_value=2.0,
            max_value=None,
            required_repetitions_in_row=5,
            not_final_value_required=True,
        ),
    ]
    detector = FinalStateDetector(final_features_values)

    assert not detector.is_final()

    config = get_game_config()
    interface = from_config(config)
    ocr_tool = OcrTool(config.ocrs, SevenSegmentsOcr)

    end_of_race_detected = False

    path_to_images = TEST_DIR / "assets/screenshots/end_of_race"
    for image_name in listdir(path_to_images):

        def mock_get_screenshot(*args, **kwargs):
            return np.array(Image.open(f"{path_to_images}/{image_name}"))

        monkeypatch.setattr(LocalScreen, "_grab_image", mock_get_screenshot)
        end_of_race_detected = detector.is_final(
            ocr_tool.perform(interface.grab_image())
        )

    assert end_of_race_detected
