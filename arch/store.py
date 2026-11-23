"""R2 storage for the archive (private bucket aqforecasts-archive).

Credentials come from the environment only: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY and R2_ARCHIVE_BUCKET.  Data files are write-once.
"""
from __future__ import annotations

import os
from pathlib import Path

REQUIRED_ENV = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ARCHIVE_BUCKET")


class AlreadyExists(Exception):
    pass


def client(env=os.environ):
    import boto3

    missing = [k for k in REQUIRED_ENV if not env.get(k)]
    if missing:
        raise RuntimeError("missing environment variables: " + ", ".join(missing))
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def put(s3, bucket: str, key: str, path, overwrite: bool = False) -> str:
    if not overwrite and exists(s3, bucket, key):
        raise AlreadyExists(key)
    s3.upload_file(str(path), bucket, key)
    return key


def list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return sorted(keys)
        token = resp.get("NextContinuationToken")


def get(s3, bucket: str, key: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(path))
    return path
