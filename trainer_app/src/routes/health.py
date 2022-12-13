from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from src.const import EnvVarsConfig, TMP_DIR
from src.schemas import TaskInfoResponse, WorkerResponse
from src.tasks.online_tasks import app
import itertools as it
import sqlite3
from logging import getLogger

logger = getLogger(__name__)
health_router = APIRouter()

def get_connection():
    return sqlite3.connect(EnvVarsConfig().celery_backend_url.split("/")[-1])

@health_router.get("/tasks/{task_id}", response_model=TaskInfoResponse)
def get_task_info(task_id: str):
    task = app.Task().AsyncResult(task_id)
    logger.info(task)
    return TaskInfoResponse(
        task_finish_time=task.date_done,
        task_name=task.name,
        task_id=str(task.id),
        status=task.status,
        result=None
    )


@health_router.get("/tasks", response_model=list[TaskInfoResponse])
def get_tasks_infos(connection=Depends(get_connection)):
    i = app.control.inspect()
    info_list = []
    methods = [i.active, i.scheduled]
    for meth in methods:
        result = meth.__call__()
        if not result:
            continue 
        logger.info(result)
        for t in it.chain.from_iterable(list(result.values())):
            task_info = TaskInfoResponse(
                task_finish_time=None,
                task_name=t["name"],
                task_id=t["id"],
                status=meth.__name__,
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
