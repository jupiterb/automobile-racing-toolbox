import pytest
from pynput.keyboard import Key
from vgamepad import XUSB_BUTTON
from racing_toolbox.interface.models import GamepadControl
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.observation.utils.ocr import OcrConfiguration
from racing_toolbox.environment.config import (
    EnvConfig,
    ActionConfig,
    ObservationConfig,
    RewardConfig,
)


@pytest.fixture
def game_conf():
    return GameConfiguration(
        game_id="trackmania",
        process_name="Trackmania Nations Forever",
        window_size=(800, 1000),
        discrete_actions_mapping={
            "FORWARD": Key.up,
            "BREAK": Key.down,
            "RIGHT": Key.right,
            "LEFT": Key.left,
        },
        continous_actions_mapping={
            "FORWARD": XUSB_BUTTON.XUSB_GAMEPAD_A,
            "BREAK": XUSB_BUTTON.XUSB_GAMEPAD_B,
            "DIRECT_X": GamepadControl.LEFT_JOYSTICK_X,
            "DIRECT_Y": GamepadControl.LEFT_JOYSTICK_Y,
        },
        reset_seconds=3,
        reset_keys_sequence=[Key.enter],
        reset_gamepad_sequence=[XUSB_BUTTON.XUSB_GAMEPAD_X],
        frequency_per_second=8,
        ocrs={
            "speed": (
                ScreenFrame(top=0.945, bottom=0.9875, left=0.918, right=0.9825),
                OcrConfiguration(
                    threshold=190,
                    max_digits=3,
                    segemnts_definitions={
                        0: ScreenFrame(top=0, bottom=0.09, left=0.42, right=0.60),
                        1: ScreenFrame(top=0.15, bottom=0.28, left=0.14, right=0.28),
                        2: ScreenFrame(top=0.15, bottom=0.28, left=0.85, right=1.0),
                        3: ScreenFrame(top=0.38, bottom=0.5, left=0.42, right=0.60),
                        4: ScreenFrame(top=0.58, bottom=0.73, left=0.14, right=0.28),
                        5: ScreenFrame(top=0.58, bottom=0.73, left=0.85, right=1.0),
                        6: ScreenFrame(top=0.82, bottom=0.94, left=0.42, right=0.60),
                    },
                ),
            )
        },
    )


@pytest.fixture
def env_config():
    action_config = ActionConfig(
        available_actions={
            "FORWARD": {0, 1, 2},
            "BREAK": set(),
            "RIGHT": {1, 3},
            "LEFT": {2, 4},
        }
    )

    reward_conf = RewardConfig(
        speed_diff_thresh=3,
        memory_length=2,
        speed_diff_trans=lambda x: float(x) ** 1.2,
        off_track_reward_trans=lambda reward: -abs(reward) - 100,
        clip_range=(-300, 300),
        baseline=20,
        scale=300,
    )

    observation_conf = ObservationConfig(
        frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
        shape=(84, 84),
        stack_size=4,
        lidar_config=None,
        track_segmentation_config=None,
    )

    return EnvConfig(
        action_config=action_config,
        reward_config=reward_conf,
        observation_config=observation_conf,
        max_episode_length=1_000,
    )
