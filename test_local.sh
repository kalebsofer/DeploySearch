#!/bin/bash

echo "Building container..."
docker-compose -f docker-compose.dev.yml build

echo "Starting container..."
docker-compose -f docker-compose.dev.yml up -d

echo "Waiting for service to start..."
sleep 10

echo "Running tests..."
python frontend/tests/test_api.py

echo "Cleaning up..."
docker-compose -f docker-compose.dev.yml down