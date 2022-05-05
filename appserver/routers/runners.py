from fastapi import APIRouter, Response, status
from schemas import Game, GameSystemConfiguration, GameGlobalConfiguration
from enviroments.car_racing.runner import Runner
from multiprocessing import Process
from typing import Optional, Union


runner_router = APIRouter(prefix="/games/{game_id}", tags=["runners"])


def run_carracing(fps):
    r = Runner()
    r.run(fps)


TARGETS = {"carracing": run_carracing}
processes = {}


@runner_router.get("/run")
async def run(game_id: str, response: Response, fps: int = 30) -> Optional[str]:
    if game_id in processes:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return f"Runner of id {game_id} has been already created"
    if game_id.lower() not in TARGETS:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return f"No Runner exist for {game_id}"

    runner = Process(target=TARGETS[game_id.lower()], args=[fps])
    processes[game_id.lower()] = runner
    runner.start()


@runner_router.get("/stop")
async def stop(game_id: str, response: Response) -> Optional[str]:
    if not game_id.lower() in processes:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return f"There is not active runner of id {game_id}"
    processes.pop(game_id).kill()
