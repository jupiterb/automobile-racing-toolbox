import pytest
from pydantic import BaseModel
from pynput.keyboard import Key
from vgamepad import XUSB_BUTTON
import copy
from racing_toolbox.interface.models import (
    GamepadControl,
    ScreenFrame,
)
from racing_toolbox.interface.config import GameConfiguration, OcrConfiguration
from racing_toolbox.environment.config import RewardConfig, ObservationConfig, EnvConfig
from racing_toolbox.observation.config import TrackSegmentationConfig, LidarConfig
from racing_toolbox.training.config import (
    TrainingConfig,
    ModelConfig,
    DQNConfig,
    ReplayBufferConfig,
)


DEFAULT_CONFIGS: dict[type[BaseModel], BaseModel] = {
    GameConfiguration: GameConfiguration(
        game_id="trackmania",
        process_name="Trackmania Nations Forever",
        window_size=(800, 1000),
        observation_frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
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
    ),
    EnvConfig: EnvConfig(
        reward_config=RewardConfig(
            speed_diff_thresh=3,
            memory_length=2,
            speed_diff_trans=lambda x: float(x) ** 1.2,
            off_track_reward_trans=lambda reward: -abs(reward) - 100,
            clip_range=(-300, 300),
            baseline=20,
            scale=300,
        ),
        observation_config=ObservationConfig(
            shape=(84, 84),
            stack_size=4,
            lidar_config=LidarConfig(
                depth=3,
                angles_range=(-90, 90, 10),
                lidar_start=(0.9, 0.5),
            ),
            track_segmentation_config=TrackSegmentationConfig(
                track_color=(200, 200, 200),
                tolerance=80,
                noise_reduction=5,
            ),
        ),
        max_episode_length=1_000,
    ),
    TrainingConfig: TrainingConfig(
        num_rollout_workers=2,
        rollout_fragment_length=10,
        train_batch_size=12,
        max_iterations=122,
        algorithm=DQNConfig(
            v_min=-100,
            v_max=100,
            replay_buffer_config=ReplayBufferConfig(capacity=50_000),
        ),
        model=ModelConfig(
            fcnet_hiddens=[100, 256],
            fcnet_activation="relu",
            conv_filters=[
                (32, 8, 4),
                (64, 4, 2),
                (64, 3, 1),
                (64, 11, 1),
            ],
        ),
    ),
}


@pytest.fixture
def game_conf():
    return copy.deepcopy(DEFAULT_CONFIGS[GameConfiguration])


@pytest.fixture
def config(request):
    cls: type[BaseModel] = request.param
    assert cls in DEFAULT_CONFIGS, f"No config testcase defined for {cls}"
    return copy.deepcopy(DEFAULT_CONFIGS[cls])