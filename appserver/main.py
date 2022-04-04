from uuid import UUID
from fastapi import FastAPI
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


@app.get("/games/{id}/trainings")
async def get_trainings(id) -> List[Training]:
    return repo.get_trainings(UUID(id))


@app.get("/games/{id}/agents")
async def get_agents(id) -> List[Agent]:
    return repo.get_agents(UUID(id))
