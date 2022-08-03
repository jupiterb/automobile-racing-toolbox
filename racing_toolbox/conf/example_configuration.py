from pynput.keyboard import Key
from interface.config import (
    GameConfiguration,
    SteeringAction,
    ScreenFrame,
    OcrConfiguration,
    NoResultPolicy,
)
from rl.config import EventDetectionParameters


common_ocr_segments_coordinates = {
    0: (0.14, 0.5),
    1: (0.3, 0.15),
    2: (0.32, 0.5),
    3: (0.3, 0.85),
    4: (0.5, 0.5),
    5: (0.7, 0.15),
    6: (0.68, 0.5),
    7: (0.7, 0.85),
    8: (0.85, 0.5),
}


def get_game_config() -> GameConfiguration:
    return GameConfiguration(
        game_id="trackmania",
        process_name="Trackmania Nations Forever",
        window_size=(1000, 800),
        obervation_frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
        discrete_actions_mapping={
            SteeringAction.FORWARD: Key.up,
            SteeringAction.BREAK: Key.down,
            SteeringAction.RIGHT: Key.right,
            SteeringAction.LEFT: Key.left,
        },
        reset_seconds=3,
        reset_keys_sequence=[Key.enter],
        frequency_per_second=10,
        ocrs={
            "speed": (
                ScreenFrame(top=0.94, bottom=0.99, left=0.91, right=0.99),
                OcrConfiguration(
                    threshold=200,
                    segments_coordinates=common_ocr_segments_coordinates,
                    no_result_policy=NoResultPolicy.RETURN_LAST,
                    no_elements_policy=NoResultPolicy.RETURN_ZERO,
                    try_lower_threshold=True,
                    debug=True,
                ),
            ),
            "checkpoint_sec": (
                ScreenFrame(top=0.28, bottom=0.32, left=0.475, right=0.503),
                OcrConfiguration(
                    threshold=200,
                    segments_coordinates=common_ocr_segments_coordinates,
                    no_result_policy=NoResultPolicy.RETURN_NEGATIVE,
                    no_elements_policy=NoResultPolicy.RETURN_NEGATIVE,
                    try_lower_threshold=False,
                    debug=False,
                ),
            ),
            "checkpoint_ms": (
                ScreenFrame(top=0.28, bottom=0.32, left=0.51, right=0.54),
                OcrConfiguration(
                    threshold=200,
                    segments_coordinates=common_ocr_segments_coordinates,
                    no_result_policy=NoResultPolicy.RETURN_NEGATIVE,
                    no_elements_policy=NoResultPolicy.RETURN_NEGATIVE,
                    try_lower_threshold=False,
                    debug=False,
                ),
            ),
        },
    )


def get_final_state_detection_parameters() -> list[EventDetectionParameters]:
    return [
        EventDetectionParameters(
            feature_name="speed",
            min_value=float("-inf"),
            max_value=2.0,
            required_repetitions_in_row=5,
            different_values_required=5,
        )
    ]


def get_checkpoint_detection_parameters() -> list[EventDetectionParameters]:
    return [
        EventDetectionParameters(
            feature_name="checkpoint_ms",
            min_value=0.0,
            max_value=99.0,
            required_repetitions_in_row=3,
            different_values_required=5,
        ),
        EventDetectionParameters(
            feature_name="checkpoint_sec",
            min_value=0.0,
            max_value=59.0,
            required_repetitions_in_row=3,
            different_values_required=5,
        ),
    ]
