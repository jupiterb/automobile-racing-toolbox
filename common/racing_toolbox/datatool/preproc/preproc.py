import logging

from racing_toolbox.datatool.utils import DatasetBasedEnv
from racing_toolbox.environment.builder import wrapp_env
from racing_toolbox.environment.config import EnvConfig

logger = logging.getLogger(__name__)

def preprocess(env: DatasetBasedEnv, config: EnvConfig):
    wrapped_env = wrapp_env(env, config)
    done = False

    while not done:
        try:
            obs, reward, done, _ = wrapped_env.step(None)
            try:
                action = wrapped_env.reverse_action(wrapped_env.last_action)
            except AttributeError:
                # in case any ActionWrapper wasn't applied
                action = wrapped_env.last_action
            except Exception as e:
                logger.warn(f"Exception while preprocessing: {e}")
                continue
            yield obs, reward, action, done
        except AssertionError:
            # needed for make FrameStack wrapper working
            continue
