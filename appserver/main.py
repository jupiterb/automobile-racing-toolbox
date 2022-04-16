from fastapi import FastAPI, status, Response, Request
from fastapi.responses import PlainTextResponse

from schemas import Game, Training
from repository import InMemoryRepository, GuardRepository
from training import TrainingManager
from utils.custom_exceptions import ItemNotFound


app = FastAPI()

games = InMemoryRepository[str, Game]()

trainings = GuardRepository[str, tuple[str, str], Training](
    InMemoryRepository[tuple[str, str], Training](),
    games,
    lambda game_training_id: game_training_id[0]
)

training_manager = TrainingManager()


@app.exception_handler(ItemNotFound)
async def handle_item_not_found(request: Request, exception: ItemNotFound):
    return PlainTextResponse(f"Resource {exception.item_name} not found", status_code=status.HTTP_404_NOT_FOUND)


@app.get("/")
async def root():
    return {"Hello": "World"}


# API for games


@app.get("/games")
async def get_games() -> list[Game]:
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
    trainings.delete_when(lambda game_training_id: game_training_id[0] == game_id)
    games.delete_item(game_id)


# API for trainings


@app.get("/games/{game_id}/trainings")
async def get_trainings(game_id: str) -> list[Training]:
    return trainings.get_all(lambda game_training_id: game_training_id[0] == game_id)


@app.get("/games/{game_id}/trainings/{training_id}")
async def get_training(game_id: str, training_id: str) -> Training:
    return trainings.get_item((game_id, training_id))


@app.post("/games/{game_id}/trainings/{training_id}", status_code=status.HTTP_201_CREATED)
async def add_training(game_id: str, training_id: str, description: str, response: Response) -> Training:
    new_training = Training(id=game_id, description=description)
    created, returned_training = trainings.add_item((game_id, training_id), new_training)
    if not created:
        response.status_code = status.HTTP_200_OK
    return returned_training
    

@app.delete("/games/{game_id}/trainings/{training_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training(game_id: str, training_id: str):
    trainings.delete_item((game_id, training_id))


@app.get("/games/{game_id}/trainings/{training_id}/run", status_code=status.HTTP_204_NO_CONTENT)
async def run_training(game_id: str, training_id: str):
    game = games.get_item(game_id)
    training = trainings.get_item((game_id, training_id))
    training_manager.run_training(
        game.system_configuration,
        game.global_configuration,
        training.parameters
    )


@app.get("/games/{game_id}/trainings/{training_id}/stop")
async def stop_training(game_id: str, training_id: str) -> Training:
    result = training_manager.stop_training()
    return trainings.update_item((game_id, training_id), result=result)
