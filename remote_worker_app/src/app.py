from fastapi import FastAPI
from remote_worker_app.src.route import router

app = FastAPI()
app.include_router(router)
