from fastapi import FastAPI, status, Response
from typing import List

from schemas import Game, Training
from repository import AbstractRepository, InMemoryRepository


app = FastAPI()

repo: AbstractRepository = InMemoryRepository()


@app.get("/")
async def root():
    return {"Hello": "World"}


# API for games


@app.get("/games")
async def get_games() -> List[Game]:
    return repo.view_games()


@app.post("/games/{game_id}", status_code=status.HTTP_201_CREATED)
async def add_game(game_id: str, description: str, response: Response) -> Game:
    created, game = repo.add_game(game_id, description)
    if not created:
        response.status_code = status.HTTP_200_OK
    return game


@app.delete("/games/{game_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_game(game_id: str):
    repo.delete_game(game_id)


# API for trainings


@app.get("/games/{game_id}/trainings")
async def get_trainings(game_id: str) -> List[Training]:
    return repo.view_trainings(game_id)


@app.get("/games/{game_id}/trainings/{training_id}")
async def get_training(game_id: str, training_id: str):
    return repo.view_training(game_id, training_id)


@app.post("/games/{game_id}/trainings/{training_id}", status_code=status.HTTP_201_CREATED)
async def add_training(game_id: str, training_id: str, description: str, response: Response) -> Training:
    created, training = repo.create_training(game_id, training_id, description)
    if not created:
        response.status_code = status.HTTP_200_OK
    return training
    

@app.delete("/games/{game_id}/trainings/{training_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training(game_id: str, training_id: str):
    repo.delete_training(game_id, training_id)


@app.get("/games/{game_id}/trainings/{training_id}/run", status_code=status.HTTP_204_NO_CONTENT)
async def run_training(game_id: str, training_id: str):
    pass


@app.get("/games/{game_id}/trainings/{training_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_training(game_id: str, training_id: str, response: Response):
    pass
