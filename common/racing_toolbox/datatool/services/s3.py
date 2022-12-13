import os
import tables as tb
from pathlib import Path
from contextlib import contextmanager
from typing import Generator
import boto3
from racing_toolbox.datatool.datasets import Dataset, DatasetModel
import uuid, shutil


class S3Dataset(Dataset):
    def __init__(
        self, bucket_name: str, file_ref: str, aws_key: str, aws_secret_key: str
    ) -> None:
        session = boto3.Session(aws_key, aws_secret_key)
        self.key = file_ref
        self.bucket = session.resource("s3").Bucket(bucket_name)
        self.client = session.client("s3")

    @contextmanager
    def get(self) -> Generator[DatasetModel, None, None]:
        with temp_dir() as tmpdir:
            fname = "tmp_h5.h5"
            path = tmpdir / fname
            with open(path, 'wb') as f:
                self.client.download_fileobj('automobile-training-test', self.key, f)
            
            with tb.File(path, driver="H5FD_CORE") as file:
                yield DatasetModel(
                    fps=int(file.root.fps[0]),
                    observations=file.root.observations,
                    actions=file.root.actions,
                )


# cannot use tempfile lib, because pytables cannot work with BytesIO
@contextmanager
def temp_dir() -> Generator[Path, None, None]:
    dir = Path(str(uuid.uuid1()))
    dir.mkdir()
    yield dir 
    shutil.rmtree(str(dir.absolute()))
