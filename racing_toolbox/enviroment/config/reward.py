from typing import Callable
from pydantic import BaseModel, PositiveFloat, PositiveInt, Field


class RewardConfig(BaseModel):
    # minimal threshold of speed drop/increase to apply additional punsihment or reward
    speed_diff_thresh: PositiveInt

    # how much frames should be taken into consideration in calculating mean speed to determine deviation
    memory_length: PositiveInt

    # function to be applied to the speed difference, to create reward boost/punishment
    speed_diff_trans: Callable[[float], float] = Field(
        default=lambda x: float(x) ** 1.2, exclude=True
    )

    # function to be applied to the reward when off-track
    off_track_reward_trans: Callable[[float], float] = Field(
        default=lambda reward: -abs(reward) - 100, exclude=True
    )

    # wether to termiante when offtrack driving is detected
    off_track_termination: bool = True

    # range to which reward will be clipped
    clip_range: tuple[float, float]

    # reward = (reward - baseline) / scale
    baseline: float
    scale: PositiveFloat
