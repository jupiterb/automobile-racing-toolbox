import boto3

from src.utils.const import EnvVarsConfig
from src.access.access_manager import AccessManager
from src.access.tokens_factory import TokensFactory
from src.aws.s3_manager import S3BucketManager
from src.aws.iam_manager import IAMManager


_env_config = EnvVarsConfig()

_session = boto3.Session(_env_config.aws_key_id, _env_config.aws_secret_key)

s3 = S3BucketManager(_session, _env_config.bucket_name)
iam = IAMManager(_session, _env_config.access_group_name)

tokens_factory = TokensFactory(
    _env_config.redirect_uri,
    _env_config.google_client_id,
    _env_config.google_client_secret,
)

access_manager = AccessManager(iam)
