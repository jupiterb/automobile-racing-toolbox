from gym.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
    TimeLimit,
)
import gym
import wandb
from racing_toolbox.environment.wrappers import observation, stats, action, reward
from racing_toolbox.environment.config import (
    ActionConfig,
    RewardConfig,
    ObservationConfig,
    FinalValueDetectionParameters,
)
from racing_toolbox.interface import controllers
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.final_state.detector import FinalStateDetector
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface import from_config
from racing_toolbox.observation.utils.ocr import OcrTool, SevenSegmentsOcr
from racing_toolbox.observation.vae import load_vae_from_wandb_checkpoint


def setup_env(game_config: GameConfiguration, env_config: EnvConfig) -> gym.Env:
    interface = (
        from_config(game_config, controllers.KeyboardController)
        if env_config.action_config.available_actions
        else from_config(game_config, controllers.GamepadController)
    )

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
        env = action.DiscreteActionToVectorWrapper(env, config.available_actions)
    return env


def reward_wrappers(env: gym.Env, config: RewardConfig) -> gym.Env:
    env = reward.SpeedDropPunishment(
        env, config.memory_length, config.speed_diff_thresh, config.speed_diff_exponent
    )
    env = reward.OffTrackPunishment(
        env,
        off_track_reward=config.off_track_reward,
        terminate=config.off_track_termination,
    )
    env = reward.ClipReward(env, *config.clip_range)
    env = reward.StandarizeReward(env, config.baseline, config.scale)
    return env


def observation_wrappers(
    env: gym.Env, config: ObservationConfig, video_freq, video_len
) -> gym.Env:
    if not config.vae_config:
        env = observation.CutImageWrapper(env, config.frame)
        if wandb.run is not None:
            env = observation.WandbVideoLogger(env, video_freq, video_len)

    if config.vae_config:
        vae, frame = load_vae_from_wandb_checkpoint(
            config.vae_config.wandb_checkpoint_ref, return_frame=True
        )
        env = observation.CutImageWrapper(env, frame)
        if wandb.run is not None:
            env = observation.WandbVideoLogger(env, video_freq, video_len)
        env = observation.VaeObservationWrapper(env, vae=vae)
        if wandb.run is not None:
            env = observation.VaeVideoLogger(
                env, 200_000, 200, vae=vae, decode_only=True
            )
    elif config.track_segmentation_config:
        env = observation.TrackSegmentationWrapper(
            env, config.track_segmentation_config
        )
        if config.lidar_config:
            env = observation.LidarWrapper(env, config.lidar_config)
        else:
            env = ResizeObservation(env, config.shape)
    else:
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, config.shape)
        env = observation.RescaleWrapper(env)
        env = observation.SqueezingWrapper(env)

    if config.observe_speed:
        env = observation.SpeedAppendingWrapper(env, 260)
    
    if config.stack_size >= 1:
        env = FrameStack(env, config.stack_size)
    return env
