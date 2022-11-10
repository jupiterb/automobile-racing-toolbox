from fastapi import FastAPI
from racing_toolbox.training.worker.route import router

app = FastAPI()
app.include_router(router)
