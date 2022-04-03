from unicodedata import name
from fastapi import FastAPI
from typing import List
from uuid import UUID, uuid4

from models import GameEnviroment, GameGlobalConfiguration, GameSystemConfiguration, Agent


app = FastAPI()

db: List[GameEnviroment] = [
    GameEnviroment( 
        id=uuid4(), 
        name="Trackmania Nations Forever",
        agents=[
            Agent(id=uuid4(), name="Zygzak McQueen")
        ], 
        trainings=[],
        global_configuration=GameGlobalConfiguration(control_actions = {}),
        system_configuration=GameSystemConfiguration(path_to_geame_exe = "abc")
    ),
    GameEnviroment( 
        id=uuid4(), 
        name="Forza Motorsport 5",
        agents=[], 
        trainings=[],
        global_configuration=GameGlobalConfiguration(control_actions = {}),
        system_configuration=GameSystemConfiguration(path_to_geame_exe = "xyz")
    )
]

@app.get("/")
async def root():
    return {"Hello": "World"}


@app.get("/games")
async def get_games():
    return db


@app.get("/games/{id}")
async def get_game(id):
    uuid = UUID(id)
    return next(game for game in db if game.id == uuid)


@app.get("/games/{id}/agents")
async def get_agents(id):
    uuid = UUID(id)
    return next(game for game in db if game.id == uuid).agents


