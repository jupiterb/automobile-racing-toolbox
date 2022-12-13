from pathlib import Path
from dotenv import load_dotenv

ROOT_PATH = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_PATH / "resources/.env"
load_dotenv(ENV_PATH)

from fastapi import FastAPI
from src.routes import online_router, health_router, registry_router, offline_router

app = FastAPI(version="1.0.0")
app.include_router(online_router)
app.include_router(offline_router)
app.include_router(health_router)
app.include_router(registry_router)
