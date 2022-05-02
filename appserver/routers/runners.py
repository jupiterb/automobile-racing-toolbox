from fastapi import APIRouter, Response, status
from schemas import Game, GameSystemConfiguration, GameGlobalConfiguration
from enviroments.car_racing.runner import Runner
from multiprocessing import Process
from typing import Union

runner_router = APIRouter(prefix="/run", tags=["runners"])

processes = {}


def run_trackmania(fps):
    r = Runner()
    r.run(fps)


@runner_router.post("/carracing/{runner_id}")
async def run_carracing(
    runner_id: Union[str, int], response: Response, fps: int = 30
) -> int:
    if runner_id in processes:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return f"Runner of id {runner_id} has been already created"
    trackmania_process = Process(target=run_trackmania, args=[fps])
    processes[runner_id] = trackmania_process
    trackmania_process.start()


@runner_router.delete("/carracing/{runner_id}")
async def stop_carracing(runner_id: Union[str, int], response: Response) -> None:
    if not runner_id in processes:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return f"There is not active runner of id {runner_id}"
    processes[runner_id].kill()
    del processes[runner_id]
