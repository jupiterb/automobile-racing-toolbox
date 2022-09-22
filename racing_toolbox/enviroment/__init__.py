from racing_toolbox.enviroment.real_time import RealTimeEnviroment
from gym.envs.registration import register
from inspect import getmodule


def as_entry_point(cls: type):
    return f"{getmodule(cls).__name__}:{cls.__name__}"


register("custom/real-time-v0", entry_point=as_entry_point(RealTimeEnviroment))
