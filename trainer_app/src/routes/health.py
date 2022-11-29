from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from trainer_app.src.schemas import TaskInfoResponse, WorkerResponse


health_router = APIRouter()


@health_router.get("/tasks/{task_id}", response_model=TaskInfoResponse)
def get_task_info(task_id: str):
    ...


@health_router.get("/tasks", response_model=list[TaskInfoResponse])
def get_tasks_infos(task_id: str):
    ...


@health_router.get("/remote_workers", response_model=list[WorkerResponse])
def get_workers(game_id: str):
    ... 


@health_router.get("/probe")
def probe():
    return PlainTextResponse("Trainer is up and running")
