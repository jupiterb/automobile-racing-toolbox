import boto3
from src.utils.schemas import UserDefinedData, UserFullData, Credentials
from src.utils.exceptions import RegistryAppException, UserExistsException


class IAMManager:
    def __init__(self, session: boto3.Session, access_group_name: str) -> None:
        self._iam_client = session.client("iam")
        self._access_group_name = access_group_name

    def add_user(self, user_data: UserDefinedData):
        if self.user_exists(user_data.username):
            raise UserExistsException(f"User {user_data.username} already exists.")
        self._create_user(user_data.username)
        try:
            self._tag_user(user_data)
            key_id, secret_key = self._create_access_key(user_data.username)
            self._tag_user_access_key(user_data.username, key_id, secret_key)
        except Exception as e:
            self.remove_user(user_data.username)
            raise e

    def get_data(self, username: str) -> UserFullData:
        if not self.user_exists(username):
            raise RegistryAppException(f"User {username} does not exist.")
        return self._get_user_data(username)

    def modify_user(
        self,
        username: str,
        new: UserDefinedData,
    ):
        if not self.user_exists(username):
            raise RegistryAppException(f"User {username} does not exist.")
        current = self._get_user_data(username)
        if not self._can_modify_user(current, new):
            raise RegistryAppException("Cannot change username or email.")
        self._tag_user(new)

    def remove_user(self, username: str):
        if self.user_exists(username):
            self._iam_client.delete_user(UserName=username)

    def user_exists(self, username: str) -> bool:
        try:
            self._iam_client.get_user(UserName=username)
            return True
        except:
            return False

    def add_user_to_access_group(self, username: str):
        self._iam_client.add_user_to_group(
            GroupName=self._access_group_name, UserName=username
        )

    def remove_user_from_access_group(self, username: str):
        self._iam_client.remove_user_from_group(
            GroupName=self._access_group_name, UserName=username
        )

    def validate_credentials(self, credentials: Credentials) -> bool:
        user_data = self._get_user_data(credentials.username)
        return user_data.password == credentials.password

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

    def _get_user_data(self, username: str) -> UserFullData:
        data = UserFullData(
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
