from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import gym
from racing_toolbox.enviroment.wrappers import *
from racing_toolbox.enviroment.config import (
    RewardConfig,
    ObservationConfig,
    FinalValueDetectionParameters,
)
from racing_toolbox.interface.training_local import TrainingLocalGameInterface
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.enviroment.final_state.detector import FinalStateDetector
from racing_toolbox.enviroment.config.env import EnvConfig


def setup_env(game_config: GameConfiguration, env_config: EnvConfig) -> gym.Env:
    interface = TrainingLocalGameInterface(game_config)
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
        final_state_detector=final_st_det,
    )
    env = reward_wrappers(env, env_config.reward_config)
    env = observation_wrappers(env, env_config.observation_config)
    env = TimeLimit(env, env_config.max_episode_length)
    return env


def reward_wrappers(env: gym.Env, config: RewardConfig) -> gym.Env:
    env = SpeedDropPunishment(
        env, config.memory_length, config.speed_diff_thresh, config.speed_diff_trans
    )
    env = OffTrackPunishment(env, metric=config.off_track_reward_trans)
    env = ClipReward(env, *config.clip_range)
    env = StandarizeReward(env, config.baseline, config.scale)
    return env


def observation_wrappers(env: gym.Env, config: ObservationConfig) -> gym.Env:
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
