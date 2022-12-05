from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from trainer_app.src.const import EnvVarsConfig, TMP_DIR
from trainer_app.src.schemas import TaskInfoResponse, WorkerResponse
from trainer_app.src.tasks.tasks import app
import itertools as it
import sqlite3

health_router = APIRouter()
connection = sqlite3.connect(EnvVarsConfig().celery_backend_url)

@health_router.get("/tasks/{task_id}", response_model=TaskInfoResponse)
def get_task_info(task_id: str):
    task = app.Task().AsyncResult(task_id)
    return TaskInfoResponse(
        task_finish_time=task.date_done,
        task_name=task.name,
        task_id=task.id,
        status=task,
        result=None
    )


@health_router.get("/tasks", response_model=list[TaskInfoResponse])
def get_tasks_infos():
    i = app.control.inspect()
    info_list = []
    for t in it.chain.from_iterable(list(i.active().values())):
        task_info = TaskInfoResponse(
            task_finish_time=None,
            task_name=t["name"],
            task_id=t["id"],
            status="PENDING",
            result=None,
        )
        info_list.append(task_info)
    cursor = connection.cursor()
    rows = cursor.execute("SELECT task_id, name, status, date_done FROM celery_taskmeta")
    for task_id, name, status, date_done in rows:
        task_info = TaskInfoResponse(
            task_finish_time=date_done,
            task_name=name,
            task_id=task_id,
            status=status,
            result=None,
        )
        info_list.append(task_info)
    return info_list


@health_router.get("/probe")
def probe():
    return PlainTextResponse("Trainer is up and running")
