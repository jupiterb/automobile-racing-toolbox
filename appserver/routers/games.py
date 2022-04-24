from fastapi import APIRouter, Response, status
from routers.common import Repositories

from schemas import Game, GameSystemConfiguration
from routers.common import Repositories


games_router = APIRouter(
    prefix="/games",
    tags=["games"]
)

games = Repositories.games
trainings = Repositories.trainings


@games_router.get("/")
async def get_games() -> list[Game]:
    return games.get_all()


@games_router.get("/{game_id}")
async def get_game(game_id: str) -> Game:
    return games.get_item(game_id)


@games_router.post("/{game_id}", status_code=status.HTTP_201_CREATED)
async def add_game(game_id: str, description: str, response: Response) -> Game:
    created, game = games.add_item(game_id, Game(id=game_id, description=description))
    if not created:
        response.status_code = status.HTTP_200_OK
    return game


@games_router.put("/{game_id}/systemconfig")
async def update_system_config(game_id: str, system_configuration: GameSystemConfiguration) -> GameSystemConfiguration:
    return games.update_item(game_id, system_configuration=system_configuration).system_configuration


@games_router.delete("/{game_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_game(game_id: str):
    trainings.delete_when(lambda game_training_id: game_training_id[0] == game_id)
    games.delete_item(game_id)
