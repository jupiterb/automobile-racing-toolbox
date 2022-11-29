from fastapi import APIRouter
from trainer_app.src.schemas import StartTaskRequest

flow_router = APIRouter()


@flow_router.put("/start")
def start_training(body: StartTaskRequest):
    """run training task"""


@flow_router.get("/stop/{task_id}")
def stop_training(task_id):
    """stop running training task"""
