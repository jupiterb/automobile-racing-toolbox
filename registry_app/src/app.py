import http
from fastapi import FastAPI, Request, Response
import logging

from src.utils.exceptions import RegistryAppException
from src.route import router
from src.dependences import access_manager

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router)


@app.exception_handler(RegistryAppException)
def registry_app_exception_handler(request: Request, e: RegistryAppException):
    logger.info(f"registry app exception: {e.message}")
    return Response(status_code=http.HTTPStatus.FORBIDDEN, content=e.message)


@app.exception_handler(Exception)
def general_exception_handler(request: Request, e: Exception):
    logger.warn(f"unidentified uerror: {e}")
    return Response(
        status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR,
        content=f"Something went wrong: {e}",
    )


@app.on_event("startup")
def startup_event():
    logger.info("startup event")
    access_manager.start_give_access()


@app.on_event("shutdown")
def shutdown_event():
    logger.info("shutdown event")
    access_manager.remove_access_everyone()
