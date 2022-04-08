from fastapi import FastAPI, status, Response
from fastapi.responses import PlainTextResponse
from typing import List

from schemas import GameEnviromentBase, TrainingBase, Training, TrainingParameters
from repository import InMemoryRepository


app = FastAPI()
repo = InMemoryRepository()


@app.get("/")
async def root():
    return {"Hello": "World"}


@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return PlainTextResponse('Resource not found', status_code=status.HTTP_404_NOT_FOUND)


@app.get("/games")
async def get_games() -> List[GameEnviromentBase]:
    return repo.view_games()


# API for trainings


@app.get("/games/{game_name}/trainings")
async def get_trainings(game_name: str) -> List[TrainingBase]:
    return repo.view_trainings(game_name)


@app.get("/games/{game_name}/trainings/{training_name}")
async def get_training(game_name: str, training_name: str):
    return repo.view_training(game_name, training_name)


@app.post("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_201_CREATED)
async def add_training(game_name: str, training_name: str, full_training_name: str, response: Response) -> Training:
    created, training = repo.create_training(game_name, TrainingBase(endpoint_name = training_name, full_name = full_training_name))
    if not created:
        response.status_code = status.HTTP_200_OK
    return training
    

@app.delete("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training(game_name: str, training_name: str):
    repo.delete_training(game_name, training_name)


@app.get("/games/{game_name}/trainings/{training_name}/run", status_code=status.HTTP_204_NO_CONTENT)
async def run_training(game_name: str, training_name: str, response: Response):
    repo.run_training(game_name, training_name)


@app.get("/games/{game_name}/trainings/{training_name}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_training(game_name: str, training_name: str, response: Response):
    repo.stop_training(game_name, training_name)


@app.post("/games/{game_name}/trainings/{training_name}")
async def update_training(game_name: str, training_name: str, training_parameters: TrainingParameters) -> Training:
    return repo.update_training(game_name, training_name, training_parameters)
