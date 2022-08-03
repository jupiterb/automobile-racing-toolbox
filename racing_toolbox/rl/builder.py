from interface import GameInterface
from rl.config import RewardConfig, ObservationConfig
from rl.wrappers import *
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import gym


def reward_wrappers(
    env: gym.Env, config: RewardConfig, game_interface: GameInterface
) -> gym.Env:
    if config.checkpoint_detector_parameters:
        env = CheckpointReward(
            env, config.checkpoint_detector_parameters, game_interface
        )
    env = SpeedDropPunishment(
        env, config.memory_length, config.speed_diff_thresh, config.speed_diff_trans
    )
    env = OffTrackPunishment(env, metric=config.off_track_reward_trans)
    env = ClipReward(env, *config.clip_range)
    return env


def observation_wrappers(env: gym.Env, config: ObservationConfig) -> gym.Env:
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, config.shape)
    env = FrameStack(env, 4)
    env = SqueezingWrapper(env)
    return env
