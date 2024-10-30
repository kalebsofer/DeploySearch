#!/bin/bash
set -e

# Start MinIO server
minio server --console-address ":9001" /data &

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until curl -sf "http://localhost:9000/minio/health/live"; do
    sleep 1
done

# Run the initialization script
echo "Initializing MinIO..."
python3 /usr/local/bin/init_minio.py

# Wait for the MinIO process
wait