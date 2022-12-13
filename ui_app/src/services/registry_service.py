import boto3
from typing import Optional

from ui_app.src.config import UserData


class RegistryService:
    def __init__(self, url: str):
        self._url = url

        self._session = boto3.Session("X", "Y")
        self._iam_client = self._session.client("iam")
        self._s3 = self._session.resource("s3").Bucket("automobile-training-test")

    def add_user(
        self, username: str, password: str, email: str, wandb_api_key: str
    ) -> bool:
        """returns True if user added, False if user with same username already exists"""
        # 0. Check user exist
        try:
            self._iam_client.get_user(UserName=username)
            return False
        except:
            pass

        # 1. Add user
        self._iam_client.create_user(UserName=username)

        # 2. Add access key
        response = self._iam_client.create_access_key(UserName=username)
        access_key = response["AccessKey"]
        key_id = access_key["AccessKeyId"]
        secret_key = access_key["SecretAccessKey"]

        # 3. Add tags
        self._iam_client.tag_user(
            UserName=username,
            Tags=[
                {"Key": "user_key_id", "Value": key_id},
                {"Key": "user_secret_key", "Value": secret_key},
                {"Key": "wandb_api_key", "Value": wandb_api_key},
                {"Key": "email", "Value": email},
                {"Key": "password", "Value": password},
            ],
        )

        # 4. Add user to group
        self._iam_client.add_user_to_group(
            GroupName="AutombileTrainingsUsers", UserName=username
        )

        # 5. Initial folder structure
        for object_ in self._s3.objects.filter(Prefix="init_user_folder/"):
            key = object_.key
            copy_source = {
                "Bucket": "automobile-training-test",
                "Key": key,
            }
            subkey = "/".join(key.split("/")[1:])
            self._s3.copy(copy_source, f"users/{username}/{subkey}")

        return True

    def get_access(self, username: str, password: str) -> Optional[UserData]:
        """returns UserData if credentials are correct"""
        data = UserData(
            username=username,
            password=password,
            wandb_api_key="",
            email="",
            user_key_id="",
            user_secret_key="",
        )
        try:
            response = self._iam_client.get_user(UserName=username)
            tags = response["User"]["Tags"]
            for tag in tags:
                if tag["Key"] == "password" and tag["Value"] != password:
                    return None
                if tag["Key"] == "email":
                    data.email = tag["Value"]
                if tag["Key"] == "user_key_id":
                    data.user_key_id = tag["Value"]
                if tag["Key"] == "user_secret_key":
                    data.user_secret_key = tag["Value"]
                if tag["Key"] == "wandb_api_key":
                    data.wandb_api_key = tag["Value"]
            return data
        except:
            return None

    def modify_user(self, current: UserData, new: UserData) -> Optional[str]:
        """returns reason if cannot modify"""
        if (
            current.username != new.username
            or current.user_key_id != new.user_key_id
            or current.user_secret_key != new.user_secret_key
        ):
            return "Cannot change username od user access key"
        self._iam_client.tag_user(
            UserName=new.username,
            Tags=[
                {"Key": "wandb_api_key", "Value": new.wandb_api_key},
                {"Key": "email", "Value": new.email},
                {"Key": "password", "Value": new.password},
            ],
        )
        return None
