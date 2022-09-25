from logging import getLogger
from numbers import Number
from typing import Any, Callable, Union
import numpy as np
import inspect
from gym.wrappers.frame_stack import LazyFrames


__TYPES_TO_LOG = (Number, str, bool)


def log_reward(logger_name: str):
    logger = getLogger(logger_name)

    def wrapper(fun: callable):
        def __inner(*args, **kwargs):
            args_to_log = [arg for arg in args if isinstance(arg, __TYPES_TO_LOG)]
            kwargs_to_log = {
                k: (v if isinstance(v, __TYPES_TO_LOG) else ...)
                for k, v in kwargs.items()
            }
            reward = fun(*args, **kwargs)
            logger.debug(
                f"{callable_name(fun, args):<30} called with {args_to_log}, {kwargs_to_log} and returned {describe_reward(reward)}"
            )
            return reward

        return __inner

    return wrapper


def log_observation(logger_name: str):
    logger = getLogger(logger_name)

    def wrapper(fun: Callable[[Any, np.ndarray], np.ndarray]):
        def __inner(obj, observation: np.ndarray):
            result = fun(obj, observation)
            logger.debug(
                f"observation after {callable_name(fun, (obj, observation)):<30}: {describe_observation(observation)}"
            )
            return result

        return __inner

    return wrapper


def callable_name(fn: Callable, args) -> str:
    return f"{args[0].__class__.__name__}.{fn.__name__}"


def describe_observation(observation: Union[np.ndarray, LazyFrames]) -> str:
    if isinstance(observation, LazyFrames):
        return "<LazyFrame>"
    return f"shape={observation.shape}, min={observation.min():.2f}, max={observation.max():.2f}"


def describe_reward(reward: int) -> str:
    return f"{reward:>10}"
