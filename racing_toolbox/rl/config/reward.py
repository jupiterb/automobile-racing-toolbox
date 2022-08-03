from typing import Callable, Optional
from pydantic import BaseModel, PositiveFloat, PositiveInt
from rl.config.event_detecion_params import EventDetectionParameters


class RewardConfig(BaseModel):
    # minimal threshold of speed drop/increase to apply additional punsihment or reward
    speed_diff_thresh: PositiveInt

    # how much frames should be taken into consideration in calculating mean speed to determine deviation
    memory_length: PositiveInt

    # function to be applied to the speed difference, to create reward boost/punishment
    speed_diff_trans: Callable[[float], float]

    # function to be applied to the reward when off-track
    off_track_reward_trans: Callable[[float], float]

    # range to which reward will be clipped
    clip_range: tuple[float, float]

    # reward = (reward - baseline) / scale
    baseline: float
    scale: PositiveFloat

    # checpoint reward
    checkpoint_detector_parameters: Optional[list[EventDetectionParameters]]
