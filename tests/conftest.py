import pytest
from pydantic import BaseModel
from pynput.keyboard import Key
from vgamepad import XUSB_BUTTON
from racing_toolbox.interface.models import GamepadControl
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.interface.models.gamepad_action import GamepadAction
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.observation.utils.ocr import OcrConfiguration, OcrToolConfiguration
from racing_toolbox.environment.config import (
    EnvConfig,
    ActionConfig,
    ObservationConfig,
    RewardConfig,
)
import copy
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import RewardConfig, ObservationConfig, EnvConfig
from racing_toolbox.observation.config import TrackSegmentationConfig, LidarConfig
from racing_toolbox.training.config import (
    TrainingConfig,
    ModelConfig,
    DQNConfig,
    ReplayBufferConfig,
    EvalConfig,
)


DEFAULT_CONFIGS: dict[type[BaseModel], BaseModel] = {
    GameConfiguration: GameConfiguration(
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
            "Y": GamepadControl.AXIS_Z,
            "X": GamepadControl.AXIS_X_LEFT,
        },
        reset_seconds=3,
        reset_keys_sequence=[Key.enter],
        reset_gamepad_sequence=[XUSB_BUTTON.XUSB_GAMEPAD_X],
        frequency_per_second=8,
        ocrs=OcrToolConfiguration(
            instances={
                "speed": (
                    ScreenFrame(top=0.945, bottom=0.9875, left=0.918, right=0.9825),
                    OcrConfiguration(
                        threshold=190,
                        max_digits=3,
                        segemnts_definitions={
                            0: ScreenFrame(top=0, bottom=0.09, left=0.42, right=0.60),
                            1: ScreenFrame(
                                top=0.15, bottom=0.28, left=0.14, right=0.28
                            ),
                            2: ScreenFrame(top=0.15, bottom=0.28, left=0.85, right=1.0),
                            3: ScreenFrame(top=0.38, bottom=0.5, left=0.42, right=0.60),
                            4: ScreenFrame(
                                top=0.58, bottom=0.73, left=0.14, right=0.28
                            ),
                            5: ScreenFrame(top=0.58, bottom=0.73, left=0.85, right=1.0),
                            6: ScreenFrame(
                                top=0.82, bottom=0.94, left=0.42, right=0.60
                            ),
                        },
                    ),
                )
            },
        ),
    ),
    EnvConfig: EnvConfig(
        reward_config=RewardConfig(
            speed_diff_thresh=3,
            memory_length=2,
            speed_diff_exponent=1.2,
            off_track_reward=-100,
            clip_range=(-300, 300),
            baseline=20,
            scale=300,
        ),
        observation_config=ObservationConfig(
            frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
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
        action_config=ActionConfig(
            available_actions={
                "FORWARD": {0, 1, 2},
                "BREAK": set(),
                "RIGHT": {1, 3},
                "LEFT": {2, 4},
            }
        ),
    ),
    TrainingConfig: TrainingConfig(
        num_rollout_workers=0,
        rollout_fragment_length=10,
        train_batch_size=12,
        max_iterations=2,
        algorithm=DQNConfig(
            v_min=-100,
            v_max=100,
            replay_buffer_config=ReplayBufferConfig(capacity=100),
        ),
        model=ModelConfig(
            fcnet_hiddens=[50],
            fcnet_activation="relu",
            conv_filters=[
                (32, (8, 8), 4),
                (64, (4, 4), 2),
                (64, (3, 3), 1),
                (64, (11, 11), 1),
            ],
        ),
    ),
    EvalConfig: EvalConfig(
        eval_name="test_evaluation",
        eval_interval_frequency=1,
        eval_duration_unit="timesteps",
        eval_duration=10,
    ),
}


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
        speed_diff_exponent=1.2,
        off_track_reward=-100,
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


@pytest.fixture
def game_conf():
    return copy.deepcopy(DEFAULT_CONFIGS[GameConfiguration])


@pytest.fixture()
def training_config():
    return copy.deepcopy(DEFAULT_CONFIGS[TrainingConfig])


@pytest.fixture
def config(request):
    cls: type[BaseModel] = request.param
    assert cls in DEFAULT_CONFIGS, f"No config testcase defined for {cls}"
    return copy.deepcopy(DEFAULT_CONFIGS[cls])


@pytest.fixture
def eval_conf():
    return copy.deepcopy(DEFAULT_CONFIGS[EvalConfig])
