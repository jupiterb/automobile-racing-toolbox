from fastapi import FastAPI, status, Response
from typing import List

from schemas import GameEnviromentBase, TrainingBase, Training, TrainingParameters
from repository import Repository


app = FastAPI()
repo = Repository()


@app.get("/")
async def root():
    return {"Hello": "World"}


@app.get("/games")
async def get_games() -> List[GameEnviromentBase]:
    pass


# API for trainings


@app.get("/games/{game_name}/trainings")
async def get_trainings(game_name: str) -> List[TrainingBase]:
    pass


@app.post("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_201_CREATED)
async def add_training(game_name: str, training_name: str, full_training_name: str,  response: Response) -> Training:
    pass
    

@app.delete("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training(game_name: str, training_name: str):
   pass


@app.get("/games/{game_name}/trainings/{training_name}/run", status_code=status.HTTP_204_NO_CONTENT)
async def run_training(game_name: str, training_name: str):
    pass


@app.get("/games/{game_name}/trainings/{training_name}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_training(game_name: str, training_name: str):
    pass


@app.patch("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_204_NO_CONTENT)
async def update_training(game_name: str, training_name: str, training_parameters: TrainingParameters):
    pass
