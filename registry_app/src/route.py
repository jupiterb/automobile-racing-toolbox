import http
import boto3
import logging
from fastapi import APIRouter, Response

from src.const import EnvVarsConfig
from src.schemas import Credentials, UserDefinedData, UserModificationRequest, UserData
from src.s3_manager import S3BucketManager
from src.iam_manager import IAMManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/registry")

env_config = EnvVarsConfig()

session = boto3.Session(env_config.aws_key_id, env_config.aws_secret_key)
iam = IAMManager(session)
s3 = S3BucketManager(session)


@router.post("/user")
def add_user(user_data: UserDefinedData):
    logger.info("got new user request")
    iam.add_user(user_data)
    s3.create_user_folder(user_data.username)
    return Response(status_code=http.HTTPStatus.OK)


@router.get("/access", response_model=UserData)
def get_access(credentials: Credentials):
    logger.info("got user access request")
    return iam.get_access(credentials)


@router.post("/modify")
def modify_user(modification: UserModificationRequest):
    logger.info("got user modification request")
    iam.modify_user(modification.credentials, modification.new)
    return Response(status_code=http.HTTPStatus.OK)
