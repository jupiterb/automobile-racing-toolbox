from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from trainer_app.src.worker_registry import (
    RemoteWorkerRegistry,
    RemoteWorkerRef,
    MemoryRegistry,
)
from dataclasses import asdict
import uuid

workers_registry = APIRouter(prefix="/registry")


@workers_registry.post("/")
def register_remote_worker(
    url: str = Body(),
    game_id: str = Body(),
    reigstry: RemoteWorkerRegistry = Depends(MemoryRegistry),
):
    worker_ref = RemoteWorkerRef(url, game_id)
    reigstry.register_worker(worker_ref)
    return JSONResponse(asdict(worker_ref))


@workers_registry.post("/keepalive")
def keepalive(
    worker_id: uuid.UUID = Body(),
    reigstry: RemoteWorkerRegistry = Depends(MemoryRegistry),
):
    reigstry.update_timestamp(worker_id)
