import boto3


def main():
    session = boto3.Session(
        "twoje key id", "twoje secret key"
    )
    client = session.client("s3")
    bucket = session.resource("s3").Bucket("automobile-training-test")

    prefix = "users/Piotrek/rcordings/"

    for object_ in bucket.objects.filter(Prefix=prefix):
        key = object_.key
        print(key)

        with open('FILE_NAME', 'wb') as f:
            client.download_fileobj('automobile-training-test', key, f)

if __name__ == "__main__":
    main()
