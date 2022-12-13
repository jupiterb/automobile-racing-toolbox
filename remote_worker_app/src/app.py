from fastapi import FastAPI
from src.route import router, is_available
from src.schemas import EnvVars
import requests
import logging
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
import os
import threading
import uuid

ENV_FILE = Path(__file__).absolute().parents[1] / ".env"
_RUNNING = True


def keep_sending_keepalive(worker_id, keepalive_url, timeout):
    # config = EnvVars(_env_file=str(ENV_FILE))
    with requests.Session() as session:
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=2,
                backoff_factor=1,
                allowed_methods=None,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        )
        session.mount("http://", adapter)
        body = {"worker_id": worker_id, "available": is_available()}
        try:
            response = session.post(keepalive_url, json=body)
            assert (
                response.status_code == 200
            ), f"Invalid status code {response.status_code}, {response.content}"
        except Exception as e:
            logging.error(f"Exception during keepalive ping: {e}")
            raise e
    if _RUNNING:
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
            response = session.post(config.register_url, json=body)
            assert (
                response.status_code == 200
            ), f"Invalid status code {response.status_code}, {response.content}"
        except Exception as e:
            logging.error("Cannot connect to the trainer server")
            raise
        else:
            body = response.json()
            os.environ["SELF_ID"] = body["id_"]
            keep_sending_keepalive(body["id_"], config.keepalive_url, 5)


@app.on_event("shutdown")
def stop_keepalive():
    global _RUNNING
    _RUNNING = False
