import http
import logging
from multiprocessing import Process
from typing import Optional
from fastapi import APIRouter, Response
from threading import Lock
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.worker.worker import Worker, Address

logger = logging.getLogger(__name__)

__WORKER: Optional[Worker] = None
__WORKER_PROCESS: Optional[Process] = None
# TODO: not really elegant, think about not using global var
__LOCK = (
    Lock()
)  # TODO: probably better idea is to handle lock via middleware, or custom router

router = APIRouter(prefix="/worker")


@router.post("/sync")
def load_configs(
    game_config: GameConfiguration, env_config: EnvConfig, policy_address: Address
):
    logger.info("got sync requst")
    global __WORKER, __WORKER_PROCESS
    if not __LOCK.acquire(blocking=False):
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)

    if __WORKER is not None:
        __LOCK.release()
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)

    __WORKER = Worker(
        policy_address=policy_address,
        game_conf=game_config,
        env_conf=env_config,
    )
    __LOCK.release()


@router.get("/start")
def start_worker():
    logger.info("got start request")
    global __WORKER, __WORKER_PROCESS
    if not __WORKER:
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)
    __LOCK.acquire()
    __WORKER_PROCESS = Process(target=__WORKER.run)
    __WORKER_PROCESS.start()
    __LOCK.release()


@router.post("/stop")
def stop_worker():
    logger.info("got stop request")
    global __WORKER, __WORKER_PROCESS
    if not __WORKER_PROCESS:
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)
    __LOCK.acquire()
    __WORKER_PROCESS.kill()
    __WORKER_PROCESS.join()
    __WORKER_PROCESS = None
    __WORKER = None
    __LOCK.release()
