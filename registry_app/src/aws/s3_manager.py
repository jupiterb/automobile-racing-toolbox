import boto3


class S3BucketManager:
    def __init__(self, session: boto3.Session, bucket_name: str) -> None:
        self._bucket_name = bucket_name
        self._s3 = session.resource("s3").Bucket(self._bucket_name)

    def create_user_folder(self, username: str):
        for object_ in self._s3.objects.filter(Prefix="init_user_folder/"):
            key = object_.key
            copy_source = {
                "Bucket": self._bucket_name,
                "Key": key,
            }
            subkey = "/".join(key.split("/")[1:])
            self._s3.copy(copy_source, f"users/{username}/{subkey}")

    def remove_user_folder(self, username: str):
        # TODO
        pass
