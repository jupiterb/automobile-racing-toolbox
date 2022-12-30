import http
import logging
from multiprocessing import Process
from typing import Optional
from fastapi import APIRouter, Response
from threading import Lock
from src.schemas import SyncRequest
from src.worker import run_worker_process
from threading import Timer
from fastapi.responses import PlainTextResponse


logger = logging.getLogger(__name__)

__WORKER_ARGS: Optional[tuple] = None
__WORKER_PROCESS: Optional[Process] = None
# TODO: not really elegant, think about not using global var
__LOCK = (
    Lock()
)  # TODO: probably better idea is to handle lock via middleware, or custom router

router = APIRouter(prefix="/worker")


def is_available() -> bool:
    with __LOCK:
        return __WORKER_ARGS is None


def __unsync():
    global __WORKER_ARGS, __WORKER_PROCESS, __LOCK
    with __LOCK:
        if __WORKER_PROCESS is None:
            __WORKER_ARGS = None


@router.post("/sync")
def load_configs(body: SyncRequest):
    logger.info("got sync requst")
    global __WORKER_ARGS, __WORKER_PROCESS, __LOCK
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
    Timer(180, __unsync).start()


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
async def probe():
    return PlainTextResponse("Up and running")
