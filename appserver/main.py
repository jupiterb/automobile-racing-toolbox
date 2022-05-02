from fastapi import FastAPI, status, Request
from fastapi.responses import PlainTextResponse

from routers import games_router, trainings_router, episodes_router, runner_router
from utils.custom_exceptions import ItemNotFound


app = FastAPI()

app.include_router(games_router)
app.include_router(trainings_router)
app.include_router(episodes_router)
app.include_router(runner_router)


@app.exception_handler(ItemNotFound)
async def handle_item_not_found(request: Request, exception: ItemNotFound):
    return PlainTextResponse(
        f"Resource {exception.item_name} not found",
        status_code=status.HTTP_404_NOT_FOUND,
    )


@app.get("/")
async def root():
    return {"Hello": "World"}
