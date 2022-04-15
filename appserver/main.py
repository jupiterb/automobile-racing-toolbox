from fastapi import FastAPI, status, Response, Request
from fastapi.responses import PlainTextResponse
from typing import List, Tuple

from schemas import Game, Training
from repository import InMemoryRepository, RepositoryGurard
from training import TrainingManager
from utils.custom_exceptions import ItemNotFound


app = FastAPI()

games = InMemoryRepository[str, Game]()

trainings = InMemoryRepository[Tuple[str, str], Training]()
trainings_guard = RepositoryGurard[str, Tuple[str, str], Training](trainings, games)

training_manager = TrainingManager()


@app.exception_handler(ItemNotFound)
async def handle_item_not_found(request: Request, exception: ItemNotFound):
    return PlainTextResponse(f"Resource {exception.item_name} not found", status_code=status.HTTP_404_NOT_FOUND)


@app.get("/")
async def root():
    return {"Hello": "World"}


# API for games


@app.get("/games")
async def get_games() -> List[Game]:
    return games.get_all()


@app.get("/games/{game_id}")
async def get_game(game_id: str,) -> Game:
    return games.get_item(game_id)


@app.post("/games/{game_id}", status_code=status.HTTP_201_CREATED)
async def add_game(game_id: str, description: str, response: Response) -> Game:
    created, game = games.add_item(game_id, Game(id=game_id, description=description))
    if not created:
        response.status_code = status.HTTP_200_OK
    return game


@app.delete("/games/{game_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_game(game_id: str):
    games.delete_item(game_id)
    trainings.delete_when(lambda game_training_id: game_training_id[0] == game_id)


# API for trainings


@app.get("/games/{game_id}/trainings")
async def get_trainings(game_id: str) -> List[Training]:
    return trainings_guard.access(game_id).get_all()


@app.get("/games/{game_id}/trainings/{training_id}")
async def get_training(game_id: str, training_id: str) -> Training:
    return trainings_guard.access(game_id).get_item((game_id, training_id))


@app.post("/games/{game_id}/trainings/{training_id}", status_code=status.HTTP_201_CREATED)
async def add_training(game_id: str, training_id: str, description: str, response: Response) -> Training:
    new_training = Training(id=game_id, description=description)
    created, returned_training = trainings_guard.access(game_id).add_item((game_id, training_id), new_training)
    if not created:
        response.status_code = status.HTTP_200_OK
    return returned_training
    

@app.delete("/games/{game_id}/trainings/{training_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training(game_id: str, training_id: str):
    trainings.delete_item((game_id, training_id))


@app.get("/games/{game_id}/trainings/{training_id}/run", status_code=status.HTTP_204_NO_CONTENT)
async def run_training(game_id: str, training_id: str):
    pass


@app.get("/games/{game_id}/trainings/{training_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_training(game_id: str, training_id: str, response: Response):
    pass
