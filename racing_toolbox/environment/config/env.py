from pydantic import BaseModel, PositiveInt

from racing_toolbox.environment.config.action import ActionConfig
from racing_toolbox.environment.config.reward import RewardConfig
from racing_toolbox.environment.config.observation import ObservationConfig


class EnvConfig(BaseModel):
    action_config: ActionConfig
    reward_config: RewardConfig
    observation_config: ObservationConfig
    max_episode_length: PositiveInt
