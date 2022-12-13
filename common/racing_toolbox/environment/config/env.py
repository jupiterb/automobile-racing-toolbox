from pydantic import BaseModel, PositiveInt

from racing_toolbox.environment.config.action import ActionConfig
from racing_toolbox.environment.config.reward import RewardConfig
from racing_toolbox.environment.config.observation import ObservationConfig


class EnvConfig(BaseModel):
    action_config: ActionConfig
    reward_config: RewardConfig
    observation_config: ObservationConfig
    max_episode_length: PositiveInt
    video_freq: int = 50_000
    video_len: int = 100
