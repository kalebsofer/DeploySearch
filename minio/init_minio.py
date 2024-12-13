from minio import Minio
import os
from pathlib import Path


def init_minio():
    """Initialize MinIO with data bucket and required files."""

    # Initialize MinIO client
    client = Minio(
        "localhost:9000",
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )

    # Create bucket
    try:
        if not client.bucket_exists("data"):
            print("Creating bucket: data")
            client.make_bucket("data")
    except Exception as e:
        print(f"Error creating bucket data: {e}")

    files_to_upload = [
        "word-vector-embeddings.model",
        "training-with-tokens.parquet",
        "doc-index-64.faiss",
        "two_tower_state_dict.pth",
    ]

    # Upload files from /data directory only if they don't exist in the bucket
    for file in files_to_upload:
        source_path = Path("/data") / file
        try:
            # Check if file exists in bucket
            client.stat_object("data", file)
            print(f"File {file} already exists in bucket")
        except:
            if source_path.exists():
                try:
                    print(f"Uploading {file} to bucket")
                    client.fput_object("data", file, str(source_path))
                    print(f"Successfully uploaded {file}")
                except Exception as e:
                    print(f"Error uploading {file}: {e}")
            else:
                print(f"Warning: Source file not found: {source_path}")


if __name__ == "__main__":
    init_minio()
