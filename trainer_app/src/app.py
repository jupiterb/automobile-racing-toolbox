from fastapi import FastAPI
from trainer_app.src.routes import flow_router, health_router

app = FastAPI()
app.include_router(flow_router)
app.include_router(health_router)
