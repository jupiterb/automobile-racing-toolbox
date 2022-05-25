from fastapi import APIRouter, Response, status
from schemas.training.training_parameters import TrainingParameters

from schemas import Training
from training import TrainingManager
from routers.common import Repositories


trainings_router = APIRouter(
    prefix="/games/{game_id}/trainings",
    tags=["trainings"]
)

games = Repositories.games
trainings = Repositories.trainings
training_manager = TrainingManager()


@trainings_router.get("/")
async def get_trainings(game_id: str) -> list[Training]:
    return trainings.get_all(lambda game_training_id: game_training_id[0] == game_id)


@trainings_router.get("/{training_id}")
async def get_training(game_id: str, training_id: str) -> Training:
    return trainings.get_item((game_id, training_id))


@trainings_router.post("/{training_id}", status_code=status.HTTP_201_CREATED)
async def add_training(game_id: str, training_id: str, description: str, params: TrainingParameters, response: Response) -> Training:
    new_training = Training(id=game_id, description=description, training_parameters=params)
    created, returned_training = trainings.add_item((game_id, training_id), new_training)
    if not created:
        response.status_code = status.HTTP_200_OK
    return returned_training
    

@trainings_router.delete("/{training_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training(game_id: str, training_id: str):
    trainings.delete_item((game_id, training_id))


@trainings_router.get("/{training_id}/run", status_code=status.HTTP_204_NO_CONTENT)
async def run_training(game_id: str, training_id: str):
    game = games.get_item(game_id)
    training = trainings.get_item((game_id, training_id))
    training_manager.run_training(
        game.system_configuration,
        game.global_configuration,
        training
    )


@trainings_router.get("/{training_id}/stop")
async def stop_training(game_id: str, training_id: str) -> Training:
    result = training_manager.stop_training()
    return trainings.update_item((game_id, training_id), result=result)
