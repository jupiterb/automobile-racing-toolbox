from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import gym
from rl.wrappers import *
from rl.config import RewardConfig, ObservationConfig, FinalValueDetectionParameters
from interface.training_local import TrainingLocalGameInterface
from interface.models.game_configuration import GameConfiguration
from rl.final_state.detector import FinalStateDetector


def setup_env(
    config: GameConfiguration, reward_conf: RewardConfig, obs_conf: ObservationConfig
) -> gym.Env:
    interface = TrainingLocalGameInterface(config)
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
    env = reward_wrappers(env, reward_conf)
    env = observation_wrappers(env, obs_conf)
    env = TimeLimit(env, config.max_episode_length)
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
