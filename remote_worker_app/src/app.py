from fastapi import FastAPI
from remote_worker_app.src.route import router
from remote_worker_app.src.schemas import EnvVars
import requests
import logging
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
import os
import threading

ENV_FILE = Path(__file__).absolute().parents[1] / ".env"


def keep_sending_keepalive(worker_id, keepalive_url, timeout):
    # config = EnvVars(_env_file=str(ENV_FILE))
    with requests.Session() as session:
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=1,
                allowed_methods=None,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        )
        session.mount("http://", adapter)
        body = {"worker_id": worker_id}
        try:
            response = session.post(keepalive_url, data=body)
            assert (
                response.status_code == 200
            ), f"Invalid status code {response.status_code}"
        except Exception as e:
            logging.error(f"Exception during keepalive ping: {e}")
    threading.Timer(
        interval=timeout,
        function=keep_sending_keepalive,
        args=(worker_id, keepalive_url, timeout),
    ).start()


app = FastAPI()
app.include_router(router)


@app.on_event("startup")
def register_in_trainer_app():
    n_retries = 4
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
        body = {"url": config.self_url, "game_id": config.game_id}
        try:
            response = session.post(config.register_url, data=body)
            assert (
                response.status_code == 200
            ), f"Invalid status code {response.status_code}"
        except Exception as e:
            logging.error("Cannot connect to the trainer server")
            raise
        else:
            body = response.json()
            os.environ["SELF_ID"] = body["id_"]
            keep_sending_keepalive(body["id_"], config.keepalive_url, 5)
