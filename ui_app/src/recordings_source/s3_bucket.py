import boto3

from ui_app.src.recordings_source.abstract import AbstractRecordingsScource


class S3BucketRecordingsSource(AbstractRecordingsScource):
    def __init__(
        self, session: boto3.Session, bucket_name: str, key_prefix: str
    ) -> None:
        self._client = session.client("s3")
        self._bucket = session.resource("s3").Bucket(bucket_name)
        self._name = bucket_name
        self._prefix = key_prefix

    def get_recordings(self) -> dict[str, str]:
        recordings = {}
        for objects in self._bucket.objects.filter(Prefix=self._prefix):
            key = objects.key
            location = self._client.get_bucket_location(Bucket=self._name)[
                "LocationConstraint"
            ]
            url = f"https://{self._name}.s3.{location}.amazonaws.com/{key}"
            name = key.split("/")[-1].split(".")[0]
            recordings[name] = url
        return recordings

    def upload_recording(self, name: str, recording):
        key = f"{self._prefix}/{name}"
        self._client.put_object(Body=recording, Bucket=self._name, Key=key)
