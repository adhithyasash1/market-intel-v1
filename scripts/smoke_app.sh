#!/bin/bash
set -e

# Smoke test for Docker container
IMAGE_NAME="market-dashboard:smoke"
CONTAINER_NAME="market-dashboard-smoke"

echo "Building Image..."
docker build -t $IMAGE_NAME .

echo "Starting Container..."
docker run -d --name $CONTAINER_NAME -p 8501:8501 $IMAGE_NAME

echo "Waiting for app to start (10s)..."
sleep 10

echo "Checking health..."
if curl --fail http://localhost:8501/_stcore/health; then
    echo "✅ App is healthy!"
    STATUS=0
else
    echo "❌ Health check failed!"
    docker logs $CONTAINER_NAME
    STATUS=1
fi

echo "Cleaning up..."
docker rm -f $CONTAINER_NAME
exit $STATUS
