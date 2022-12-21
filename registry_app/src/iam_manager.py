import boto3
from src.schemas import UserDefinedData, UserData, Credentials
from src.exceptions import RegistryAppException, UserExistsException


class IAMManager:
    def __init__(self, session: boto3.Session) -> None:
        self._iam_client = session.client("iam")

    def add_user(self, user_data: UserDefinedData):
        if self._user_exists(user_data.username):
            raise UserExistsException(f"User {user_data.username} already exists.")
        self._create_user(user_data.username)
        try:
            self._tag_user(user_data)
            key_id, secret_key = self._create_access_key(user_data.username)
            self._tag_user_access_key(user_data.username, key_id, secret_key)
            self._add_user_to_group(user_data.username)
        except Exception as e:
            self.remove_user(user_data.username)
            raise e

    def get_access(self, credentials: Credentials) -> UserData:
        if not self._user_exists(credentials.username):
            raise RegistryAppException(f"User {credentials.username} does not exist.")
        user_data = self._get_user_data(credentials.username)
        if user_data.password != credentials.password:
            raise RegistryAppException(f"Wrong password.")
        return user_data

    def modify_user(self, credentials: Credentials, new: UserDefinedData):
        if not self._user_exists(credentials.username):
            raise RegistryAppException(f"User {credentials.username} does not exist.")
        current = self._get_user_data(credentials.username)
        if not self._can_modify_user(current, new):
            raise RegistryAppException("Cannot change username or email.")
        if current.password != credentials.password:
            raise RegistryAppException(f"Wrong password.")
        self._tag_user(new)

    def remove_user(self, username: str):
        if self._user_exists(username):
            self._iam_client.delete_user(UserName=username)

    def _user_exists(self, username: str) -> bool:
        try:
            self._iam_client.get_user(UserName=username)
            return True
        except:
            return False

    def _create_user(self, username: str):
        self._iam_client.create_user(UserName=username)

    def _create_access_key(self, username: str) -> tuple[str, str]:
        response = self._iam_client.create_access_key(UserName=username)
        access_key = response["AccessKey"]
        key_id = access_key["AccessKeyId"]
        secret_key = access_key["SecretAccessKey"]
        return key_id, secret_key

    def _tag_user(self, user_data: UserDefinedData):
        self._iam_client.tag_user(
            UserName=user_data.username,
            Tags=[
                {"Key": "wandb_api_key", "Value": user_data.wandb_api_key},
                {"Key": "email", "Value": user_data.email},
                {"Key": "password", "Value": user_data.password},
            ],
        )

    def _tag_user_access_key(self, username: str, key_id: str, secret_key: str):
        self._iam_client.tag_user(
            UserName=username,
            Tags=[
                {"Key": "user_key_id", "Value": key_id},
                {"Key": "user_secret_key", "Value": secret_key},
            ],
        )

    def _add_user_to_group(self, username: str):
        self._iam_client.add_user_to_group(
            GroupName="AutombileTrainingsUsers", UserName=username
        )

    def _get_user_data(self, username: str) -> UserData:
        data = UserData(
            username=username,
            password="",
            wandb_api_key="",
            email="",
            user_key_id="",
            user_secret_key="",
        )
        response = self._iam_client.get_user(UserName=username)
        tags = response["User"]["Tags"]
        for tag in tags:
            if tag["Key"] == "password":
                data.password = tag["Value"]
            if tag["Key"] == "email":
                data.email = tag["Value"]
            if tag["Key"] == "user_key_id":
                data.user_key_id = tag["Value"]
            if tag["Key"] == "user_secret_key":
                data.user_secret_key = tag["Value"]
            if tag["Key"] == "wandb_api_key":
                data.wandb_api_key = tag["Value"]
        return data

    def _can_modify_user(self, current: UserDefinedData, new: UserDefinedData) -> bool:
        return current.username == new.username and current.email == new.email
