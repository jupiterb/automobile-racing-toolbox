import boto3
import json
from typing import Generic

from ui_app.src.config_source.abstract import (
    AbstractConfigSource,
    RacingToolboxConfiguration,
)


class S3BucketConfigSource(AbstractConfigSource, Generic[RacingToolboxConfiguration]):
    def __init__(
        self, session: boto3.Session, bucket_name: str, key_prefix: str
    ) -> None:
        self._client = session.client("s3")
        self._bucket = session.resource("s3").Bucket(bucket_name)
        self._name = bucket_name
        self._prefix = key_prefix

    def get_configs(self) -> dict[str, RacingToolboxConfiguration]:
        config_cls = self.__orig_class__.__args__[0]
        configs = {}
        for objects in self._bucket.objects.filter(Prefix=self._prefix):
            key = objects.key
            response = self._client.get_object(Bucket=self._name, Key=key)
            body = response["Body"].read().decode()
            try:
                config = config_cls(**json.loads(body))
                name = key.split("/")[-1].split(".")[0]
                configs[name] = config
            except:
                pass
        return configs

    def add_config(self, name: str, config: RacingToolboxConfiguration):
        key = f"{self._prefix}/{name}.json"
        self._client.put_object(Body=config.json(), Bucket=self._name, Key=key)
