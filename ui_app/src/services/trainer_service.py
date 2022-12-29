import json
import orjson
import requests

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig
from racing_toolbox.observation.config.vae_config import (
    VAETrainingConfig,
    VAEModelConfig,
)

from ui_app.src.utils import TaskInfo


class TrainerService:
    def __init__(self, url: str) -> None:
        self._url = url

    def start_training(
        self,
        game_config: GameConfiguration,
        env_config: EnvConfig,
        training_config: TrainingConfig,
        wandb_key: str,
    ):
        data = {
            "game_config": orjson.loads(game_config.json()),
            "env_config": orjson.loads(env_config.json()),
            "training_config": orjson.loads(training_config.json()),
            "wandb_api_key": wandb_key,
        }
        encoded_data = json.dumps(data).encode("utf-8")
        requests.put(f"{self._url}/online/start", data=encoded_data)

    def resume_training(
        self,
        training_config: TrainingConfig,
        wandb_key: str,
        wandb_run_reference: str,
        checkpoint_name: str,
    ):
        data = {
            "wandb_run_reference": wandb_run_reference,
            "wandb_api_key": wandb_key,
            "training_config": orjson.loads(training_config.json()),
            "checkpoint_name": checkpoint_name,
        }
        encoded_data = json.dumps(data).encode("utf-8")
        requests.put(f"{self._url}/online/resume", data=encoded_data)

    def start_autoencoder_training(
        self,
        wandb_api_key: str,
        training_params: VAETrainingConfig,
        encoder_config: VAEModelConfig,
        bucket_name: str,
        recordings_refs: list[str],
    ):
        data = {
            "wandb_api_key": wandb_api_key,
            "training_params": orjson.loads(training_params.json()),
            "encoder_config": orjson.loads(encoder_config.json()),
            "bucket_name": bucket_name,
            "recordings_refs": recordings_refs,
        }
        encoded_data = json.dumps(data).encode("utf-8")
        requests.post(f"{self._url}/offline/start_vae", data=encoded_data)

    def stop_training(self, task_id: str):
        requests.get(f"{self._url}/online/stop/{task_id}")

    def get_trainings_tasks(self) -> list[TaskInfo]:
        response = requests.get(f"{self._url}/tasks")
        tasks = response.json()
        return [TaskInfo(**task) for task in tasks]
