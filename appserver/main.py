from uuid import UUID
from fastapi import FastAPI, status, Response
from typing import List

from models import GameEnviroment, GameGlobalConfiguration, GameSystemConfiguration, Agent, Training
from repository import InMemoryRepository

app = FastAPI()
repo = InMemoryRepository()

@app.get("/")
async def root():
    return {"Hello": "World"}


@app.get("/games")
async def get_games() -> List[GameEnviroment]:
    return repo.games


@app.get("/games/{game_name}/trainings")
async def get_trainings(game_name: str) -> List[Training]:
    return repo.get_trainings(game_name)


@app.get("/games/{game_name}/agents")
async def get_agents(game_name: str) -> List[Agent]:
    return repo.get_agents(game_name)


@app.post("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_201_CREATED)
async def add_training(game_name: str, training_name: str, full_training_name: str,  response: Response) -> Training:
    created, training = repo.add_training(game_name, training_name, full_training_name)
    if not created:
        response.status_code = status.HTTP_200_OK
    return training
    

@app.delete("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_training(game_name: str, training_name: str):
    repo.remove_training(game_name, training_name)


@app.get("/games/{game_name}/trainings/{training_name}/run", status_code=status.HTTP_204_NO_CONTENT)
async def run_training(game_name: str, training_name: str):
    repo.run_training(game_name, training_name)


@app.get("/games/{game_name}/trainings/{training_name}/finish", status_code=status.HTTP_204_NO_CONTENT)
async def finish_training(game_name: str, training_name: str):
    repo.finish_training(game_name, training_name)


@app.patch("/games/{game_name}/trainings/{training_name}", status_code=status.HTTP_204_NO_CONTENT)
async def update_training(game_name: str, training_name: str, training_data: Training):
    repo.update_training(game_name, training_name, training_data)


@app.get("/games/{game_name}/trainings/{training_name}/agent")
async def get_agent_from_training(game_name: str, training_name: str) -> Agent:
    repo.get_agent_from_training(game_name, training_name)


@app.post("/games/{game_name}/agents/{agent_name}")
async def add_agent(game_name: str, agent_name: str, agent_data: Agent) -> Agent:
    repo.add_agent(game_name, agent_name, agent_data)


@app.get("/games/{game_name}/agents/{agent_name}/use", status_code=status.HTTP_204_NO_CONTENT)
async def use_agent(game_name: str, agent_name: str):
    repo.use_agent(game_name, agent_name)


@app.delete("/games/{game_name}/agents/{agent_name}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_training(game_name: str, agent_name: str):
    repo.remove_agent(game_name, agent_name)
