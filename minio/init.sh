#!/bin/bash
set -e

minio server --console-address ":9001" /data &

echo "Waiting for MinIO to be ready..."
until curl -sf "http://localhost:9000/minio/health/live"; do
    sleep 1
done

echo "Initializing MinIO..."
python3 /usr/local/bin/init_minio.py

wait
