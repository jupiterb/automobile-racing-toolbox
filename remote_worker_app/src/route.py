import http
import logging
from multiprocessing import Process
from typing import Optional
from fastapi import APIRouter, Response
from threading import Lock
from remote_worker_app.src.schemas import SyncRequest
from remote_worker_app.src.worker import run_worker_process


logger = logging.getLogger(__name__)

__WORKER_ARGS: Optional[tuple] = None
__WORKER_PROCESS: Optional[Process] = None
# TODO: not really elegant, think about not using global var
__LOCK = (
    Lock()
)  # TODO: probably better idea is to handle lock via middleware, or custom router

router = APIRouter(prefix="/worker")


@router.post("/sync")
def load_configs(body: SyncRequest):
    logger.info("got sync requst")
    global __WORKER_ARGS, __WORKER_PROCESS
    if not __LOCK.acquire(blocking=False):
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)

    if __WORKER_ARGS is not None:
        __LOCK.release()
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)

    __WORKER_ARGS = (
        body.policy_address,
        body.game_config,
        body.env_config,
        body.wandb_api_key,
        body.wandb_project,
        body.wandb_group,
    )
    __LOCK.release()


@router.post("/start")
def start_worker():
    logger.info("got start request")
    global __WORKER_ARGS, __WORKER_PROCESS
    if not __WORKER_ARGS:
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)
    __LOCK.acquire()
    __WORKER_PROCESS = Process(target=run_worker_process, args=__WORKER_ARGS)
    __WORKER_PROCESS.start()
    __LOCK.release()


@router.post("/stop")
def stop_worker():
    logger.info("got stop request")
    global __WORKER_ARGS, __WORKER_PROCESS
    if not __WORKER_PROCESS:
        return Response(status_code=http.HTTPStatus.SERVICE_UNAVAILABLE.value)
    __LOCK.acquire()
    __WORKER_PROCESS.kill()
    __WORKER_PROCESS.join()
    __WORKER_PROCESS = None
    __WORKER_ARGS = None
    __LOCK.release()


@router.get("/probe")
def probe():
    pass
