import gym
import numpy as np

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig

from racing_toolbox.environment.builder import observation_wrappers


class ConfigValidator:
    def __init__(self) -> None:
        self._errors: list[str] = []

    def validate_discrete_actions_compatibilty(
        self, game: GameConfiguration, env: EnvConfig
    ):
        self._catch_errors(_validate_discrete_actions_compatibilty)(game, env)

    def validate_continous_actions_compatibilty(
        self, game: GameConfiguration, env: EnvConfig
    ):
        self._catch_errors(_validate_continous_actions_compatibilty)(game, env)

    def validate_model_and_observation_space_compatibilty(
        self, game: GameConfiguration, env: EnvConfig, training: TrainingConfig
    ):
        self._catch_errors(_validate_model_and_observation_space_compatibilty)(
            game, env, training
        )

    @property
    def errors(self) -> list[str]:
        return self._errors

    def reset_errors(self):
        self._errors = []

    def _catch_errors(self, func):
        def inner(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except AssertionError as e:
                self._errors.append(str(e))

        return inner


def _validate_discrete_actions_compatibilty(game: GameConfiguration, env: EnvConfig):
    env_discrete_actions = env.action_config.available_actions
    if env_discrete_actions:
        game_discrete_actions = game.discrete_actions_mapping
        assert any(
            game_discrete_actions
        ), "You want to use discrete actions space so any discrete action was defined in game interface."
        for action_name in env_discrete_actions:
            assert (
                action_name in game_discrete_actions
            ), f"{action_name} is not defined in discrete actions mapping in interface."


def _validate_continous_actions_compatibilty(game: GameConfiguration, env: EnvConfig):
    env_discrete_actions = env.action_config.available_actions
    if not env_discrete_actions:
        game_continous_actions = game.continous_actions_mapping
        assert any(
            game_continous_actions
        ), "You want to use continous actions space so any continous action was defined in game interface."


def _validate_model_and_observation_space_compatibilty(
    game: GameConfiguration, env: EnvConfig, training: TrainingConfig
):
    class FakeEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(*game.window_size, 3),
                dtype=np.uint8,
            )

    fake = FakeEnv()
    fake = observation_wrappers(
        fake, env.observation_config, env.video_freq, env.video_freq
    )
    obs_shape = fake.observation_space.shape

    model = training.model

    assert any(model.fcnet_hiddens), "FCNet layers are required."

    if len(obs_shape) == 2:
        assert not any(
            model.conv_filters
        ), f"Convolution layer require 3D observation space (Your is {obs_shape})."
        return

    assert len(obs_shape) == 3, "Only 2D and 3D observation spaces are supported"
    assert any(
        model.conv_filters
    ), "You have 3D observation space, but you don't have convolution layer"

    optput_size = lambda input, kernel, stride: (input - kernel) / stride + 1
    is_integer = lambda x: abs(x - int(x)) < 0.0001

    for i, (channels, (kernel_x, kernel_y), stride) in enumerate(model.conv_filters):
        x, y, _ = obs_shape
        assert is_integer(x) and is_integer(
            y
        ), f"Your input to {i} convolution layer ({x}, {y}) contains non-itegers."
        assert (
            x >= kernel_x and y >= kernel_y
        ), f"Your input to {i} convolution layer is to small ({int(x)}, {int(y)}) when kernel size is ({kernel_x}, {kernel_y})."
        obs_shape = (
            optput_size(x, kernel_x, stride),
            optput_size(y, kernel_y, stride),
            channels,
        )
