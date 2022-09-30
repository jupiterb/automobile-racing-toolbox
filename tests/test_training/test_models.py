import pytest
import numpy as np
from gym.spaces import Box, Discrete
from racing_toolbox.trainer.config import MLPConfig
from racing_toolbox.trainer.models.mlp import KerasMLP


@pytest.mark.parametrize("obs_space", [Box(0, 1, (10,)), Box(0, 1, (10000,))])
@pytest.mark.parametrize("action_space", [Discrete(20), Discrete(5)])
@pytest.mark.parametrize(
    "mlp_config",
    [
        MLPConfig(hiddens=[1000, 500, 100], activations="relu"),
        MLPConfig(hiddens=[100, 100, 100], activations=["relu", "tanh", "sigmoid"]),
    ],
)
def test_model_setup(obs_space, action_space, mlp_config):
    model = KerasMLP(obs_space, action_space, mlp_config, "test_model")
    obs = np.expand_dims(obs_space.sample(), 0)
    actions: np.ndarray = model.model.predict(obs)
    assert all(a.shape == (action_space.n,) for a in actions), "invalid output shape"
