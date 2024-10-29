from minio import Minio
import os
from pathlib import Path


def init_minio():
    """Initialize MinIO with required buckets and files."""

    # Initialize MinIO client
    client = Minio(
        "localhost:9000",
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        secure=False,
    )

    # Create required buckets
    buckets = ["models", "data"]
    for bucket in buckets:
        if not client.bucket_exists(bucket):
            print(f"Creating bucket: {bucket}")
            client.make_bucket(bucket)

    # Upload model files
    model_files = {
        "models": ["word-vector-embeddings.model", "two_tower_state_dict.pth"],
        "data": ["training-with-tokens.parquet", "doc-index-64.faiss"],
    }

    # Upload files from local directories to MinIO
    for bucket, files in model_files.items():
        for file in files:
            local_path = Path("/data") / file
            if local_path.exists():
                print(f"Uploading {file} to {bucket}")
                client.fput_object(bucket, file, str(local_path))
            else:
                print(f"Warning: {file} not found in local directory")


if __name__ == "__main__":
    init_minio()
