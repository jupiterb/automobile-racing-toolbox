import http
import logging
from fastapi import APIRouter, Response

from src.utils.schemas import (
    UserDefinedData,
    UserModificationRequest,
    UserFullData,
    Token,
    GetTokenRequest,
)
from src.dependences import iam, s3, access_manager, tokens_factory
from src.utils.exceptions import RegistryAppException


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/registry")


@router.post("/user")
def add_user(user_data: UserDefinedData):
    logger.info("got new user request")
    iam.add_user(user_data)
    s3.create_user_folder(user_data.username)
    return Response(status_code=http.HTTPStatus.OK)


@router.get("/access", response_model=Token)
def get_access(request: GetTokenRequest):
    logger.info("got user access request")
    if request.credentials:
        username = request.credentials.username
        if not iam.user_exists(username):
            raise RegistryAppException(f"User {username} does not exists.")
        elif not iam.validate_credentials(request.credentials):
            raise RegistryAppException(f"Wrong credentials.")
        else:
            token = tokens_factory.generate_token(request.credentials)
    else:
        token = tokens_factory.generate_google_token(request.code)
        user_data = tokens_factory.generate_google_user_data(token)
        username = user_data.username
        if not iam.user_exists(username):
            add_user(user_data)
    access_manager.give_access(username, token)
    return token


@router.get("/data", response_model=UserFullData)
def get_data(token: Token):
    logger.info("got user data request")
    username = access_manager.validate_token(token)
    return iam.get_data(username)


@router.post("/modify")
def modify_user(modification: UserModificationRequest):
    logger.info("got user modification request")
    username = access_manager.validate_token(modification.token)
    iam.modify_user(username, modification.new)
