from fastapi import FastAPI
from remote_worker_app.src.route import router
from remote_worker_app.src.schemas import EnvVars
import requests
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path

ENV_FILE = Path(__file__).absolute().parents[1] / ".env"


def register_in_trainer_app(n_retries):
    config = EnvVars(_env_file=str(ENV_FILE))
    with requests.Session() as session:
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=n_retries,
                backoff_factor=1,
                allowed_methods=None,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        )
        session.mount("http://", adapter)
        body = {"address": config.self_url, "game_id": config.game_id}
        try:
            response = session.post(config.trainer_url, data=body)
            assert (
                response.status_code == 200
            ), f"Invalid status code {response.status_code}"
        except Exception as e:
            print("Cannot connect to the trainer server")
            raise


app = FastAPI()
app.include_router(router)

register_in_trainer_app(4)
