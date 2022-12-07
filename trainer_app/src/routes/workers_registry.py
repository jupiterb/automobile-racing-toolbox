from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from src.worker_registry import (
    RemoteWorkerRegistry,
    RemoteWorkerRef,
    MemoryRegistry,
)
from src.worker_registry.in_memory_registry import get_registry
import uuid
from typing import Any


registry_router = APIRouter(prefix="/registry")


@registry_router.post("/")
def register_remote_worker(
    url: str = Body(),
    game_id: str = Body(),
    reigstry: RemoteWorkerRegistry = Depends(get_registry),
):
    worker_ref = RemoteWorkerRef(address=url, game_id=game_id)
    reigstry.register_worker(worker_ref)
    return worker_ref


@registry_router.post("/keepalive")
def keepalive(
    worker_id: uuid.UUID = Body(),
    available: bool = Body(),
    reigstry: RemoteWorkerRegistry = Depends(get_registry),
):
    reigstry.update_timestamp(worker_id, available=available)


@registry_router.get("/")
def get_registry_route(
    reigstry: RemoteWorkerRegistry = Depends(get_registry),
):
    return {"workers": reigstry.get_active_workers()}
