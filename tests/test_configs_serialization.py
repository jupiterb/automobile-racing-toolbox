import pytest
from pydantic import BaseModel
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config import TrainingConfig


@pytest.mark.parametrize(
    "config", [GameConfiguration, EnvConfig, TrainingConfig], indirect=True
)
def test_config_can_be_converted_to_json(config):
    json_config = config.json()
    cls: BaseModel = type(config)
    config_back = cls.parse_raw(json_config)
    assert config_back == config, "configs before and after serialziation differ"
