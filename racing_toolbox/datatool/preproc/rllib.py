import os
import json

from ray.rllib.policy.sample_batch import SampleBatch
from typing import Any

from racing_toolbox.datatool.utils import DatasetBasedEnv
from racing_toolbox.datatool.exceptions import ItemExists
from racing_toolbox.datatool.preproc.preproc import preprocess
from racing_toolbox.environment.config import EnvConfig


Batch = dict[str, Any]


def make_rllib_dataset(
    env: DatasetBasedEnv,
    config: EnvConfig,
    dst_path: str,
    game: str,
    user: str,
    name: str,
):
    batch = _create_batch()
    for obs, reward, action, done in preprocess(env, config):
        _add_to_batch(batch, obs, reward, done, action)
    _save(config, batch, dst_path, game, user, name)


def _create_batch() -> Batch:
    return {
        "type": "SampleBatch",
        SampleBatch.EPS_ID: [],
        SampleBatch.OBS: [],
        SampleBatch.ACTIONS: [],
        SampleBatch.REWARDS: [],
        SampleBatch.DONES: [],
        SampleBatch.ACTION_LOGP: [],
    }


def _add_to_batch(batch: Batch, obs, reward, done, action) -> Batch:
    batch[SampleBatch.EPS_ID].append(1)
    batch[SampleBatch.OBS].append(obs.to_list())
    batch[SampleBatch.ACTIONS].append(action)
    batch[SampleBatch.REWARDS].append(reward)
    batch[SampleBatch.DONES].append(done)
    batch[SampleBatch.ACTION_LOGP].append(0.0)
    return batch


def _save(
    config: EnvConfig,
    batch: Batch,
    root: str,
    game: str,
    user: str,
    name: str,
):
    path = f"{root}/{game}/{user}/{name}"
    if os.path.exists(path):
        raise ItemExists(game, user, name)

    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    data_path = f"{path}/data.json"
    with open(data_path, "w") as f:
        json.dump(batch, f)

    config_path = f"{path}/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
