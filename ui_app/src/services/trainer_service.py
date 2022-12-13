import urllib3
import json

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig

from ui_app.src.utils import TaskInfo


class TrainerService:
    def __init__(self, url: str) -> None:
        self._http = urllib3.PoolManager()
        self._url = url

    def start_training(
        self,
        game_config: GameConfiguration,
        env_config: EnvConfig,
        training_config: TrainingConfig,
        wandb_key: str,
    ):
        data = {
            "game_config": game_config,
            "env_config": env_config,
            "training_config": training_config,
            "wandb_api_key": wandb_key,
        }
        encoded_data = json.dumps(data).encode("utf-8")
        response = self._http.request(
            "PUT",
            f"{self._url}/start",
            body=encoded_data,
            headers={"Content-Type": "application/json"},
        )

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
            "training_config": training_config,
            "checkpoint_name": checkpoint_name,
        }
        encoded_data = json.dumps(data).encode("utf-8")
        response = self._http.request(
            "PUT",
            f"{self._url}/resume",
            body=encoded_data,
            headers={"Content-Type": "application/json"},
        )

    def stop_training(self, task_id: str):
        self._http.request(
            "PUT",
            f"{self._url}/stop/{task_id}",
            headers={"Content-Type": "application/json"},
        )

    def get_trainings_tasks(self) -> list[TaskInfo]:
        # response = self._http.request(
        #    "PUT",
        #    f"{self._url}/tasks",
        #    headers={"Content-Type": "application/json"},
        # )
        return [
            TaskInfo(
                task_name="Trackmania Nations Forever 01",
                status="Running",
                task_id="AQ3-DF2-12Z",
                result=None,
                task_finish_time=None,
            )
        ]
