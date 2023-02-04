from pydantic import BaseModel, PositiveInt

from racing_toolbox.environment.config.action import ActionConfig
from racing_toolbox.environment.config.reward import RewardConfig
from racing_toolbox.environment.config.observation import ObservationConfig
from racing_toolbox.observation.config import LidarConfig
from racing_toolbox.observation.config import TrackSegmentationConfig


class EnvConfig(BaseModel):
    action_config: ActionConfig
    reward_config: RewardConfig
    observation_config: ObservationConfig
    lidar_config: LidarConfig
    track_segmentation_config: TrackSegmentationConfig
    max_episode_length: PositiveInt
    video_freq: int = 50_000
    video_len: int = 100
