from pydantic import BaseModel, PositiveFloat, PositiveInt
from typing import Optional


class SpeedDropPunishmentConfig(BaseModel):
    # minimal threshold of speed drop/increase to apply additional punsihment or reward
    speed_diff_thresh: PositiveInt

    # how much frames should be taken into consideration in calculating mean speed to determine deviation
    memory_length: PositiveInt

    # function to be applied to the speed difference, to create reward boost/punishment
    speed_diff_exponent: float = 1.2


class SafetyConfig(BaseModel):
    # safety_base = mean( sorted( rays_len )[0:shortest_rays_number] )
    shortest_rays_number: PositiveInt

    # safety = safety_base * weight
    weight: PositiveFloat = 0.5


class RewardConfig(BaseModel):
    # function to be applied to the reward when off-track
    off_track_reward: float = -1

    # wether to termiante when offtrack driving is detected
    off_track_termination: bool = True

    # range to which reward will be clipped
    clip_range: tuple[float, float]

    speed_drop_punishment_config: Optional[SpeedDropPunishmentConfig] = None

    # if None, afety = 1
    safety_config: Optional[SafetyConfig] = None

    # reward = (reward - baseline) / scale
    baseline: float
    scale: PositiveFloat
