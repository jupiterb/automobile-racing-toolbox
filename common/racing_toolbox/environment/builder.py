from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import gym
import wandb
from racing_toolbox.environment.wrappers import *
from racing_toolbox.environment.config import (
    ActionConfig,
    RewardConfig,
    ObservationConfig,
    FinalValueDetectionParameters,
)
from racing_toolbox.environment.wrappers.observation import CutImageWrapper
<<<<<<< HEAD:racing_toolbox/environment/builder.py
from racing_toolbox.interface.controllers.gamepad import GamepadController
from racing_toolbox.interface.controllers.keyboard import KeyboardController
=======
from racing_toolbox.interface import controllers
>>>>>>> 2a43dfdba2e67dea4ff8ec1fd72a61921726cc4f:common/racing_toolbox/environment/builder.py

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.final_state.detector import FinalStateDetector
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface import from_config
from racing_toolbox.observation.utils.ocr import OcrTool, SevenSegmentsOcr


def setup_env(game_config: GameConfiguration, env_config: EnvConfig) -> gym.Env:
<<<<<<< HEAD:racing_toolbox/environment/builder.py
    if env_config.action_config.available_actions is not None:
        interface = from_config(game_config, KeyboardController)
    else:
        interface = from_config(game_config, GamepadController)
=======
    interface = from_config(game_config, controllers.KeyboardController)
>>>>>>> 2a43dfdba2e67dea4ff8ec1fd72a61921726cc4f:common/racing_toolbox/environment/builder.py

    ocr_tool = OcrTool(game_config.ocrs, SevenSegmentsOcr)

    final_st_det = FinalStateDetector(
        [
            FinalValueDetectionParameters(
                feature_name="speed",
                min_value=2,
                max_value=float("inf"),
                required_repetitions_in_row=20,
                not_final_value_required=False,
            )
        ]
    )

    env = gym.make(
        "custom/real-time-v0",
        game_interface=interface,
        ocr_tool=ocr_tool,
        final_state_detector=final_st_det,
    )

    env = wrapp_env(env, env_config)
    env = TimeLimit(env, env_config.max_episode_length)
    return env


def wrapp_env(env: gym.Env, env_config: EnvConfig) -> gym.Env:
    env = action_wrappers(env, env_config.action_config)
    env = reward_wrappers(env, env_config.reward_config)
    env = observation_wrappers(
        env, env_config.observation_config, env_config.video_freq, env_config.video_len
    )
    return env


def action_wrappers(env: gym.Env, config: ActionConfig) -> gym.Env:
    if config.available_actions is not None:
        env = DiscreteActionToVectorWrapper(env, config.available_actions)
    return env


def reward_wrappers(env: gym.Env, config: RewardConfig) -> gym.Env:
    env = SpeedDropPunishment(
        env, config.memory_length, config.speed_diff_thresh, config.speed_diff_exponent
    )
    env = OffTrackPunishment(
        env,
        off_track_reward=config.off_track_reward,
        terminate=config.off_track_termination,
    )
    env = ClipReward(env, *config.clip_range)
    env = StandarizeReward(env, config.baseline, config.scale)
    return env


def observation_wrappers(
    env: gym.Env, config: ObservationConfig, video_freq, video_len
) -> gym.Env:
    env = CutImageWrapper(env, config.frame)
    if wandb.run is not None:
        env = WandbVideoLogger(env, video_freq, video_len)
    if config.track_segmentation_config:
        env = TrackSegmentationWrapper(env, config.track_segmentation_config)
        if config.lidar_config:
            env = LidarWrapper(env, config.lidar_config)
        else:
            env = ResizeObservation(env, config.shape)
    else:
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, config.shape)
        env = RescaleWrapper(env)
    env = FrameStack(env, config.stack_size)
    env = SqueezingWrapper(env)
    return env
