#!/bin/bash

# EoMT Docker Build Script
# This script builds the EoMT Docker image and sets up the environment

set -e

echo "🐋 Building EoMT Docker Image..."

# Build the Docker image
docker build -t eomt:latest .

echo "✅ Docker image built successfully!"

echo "📋 Setup complete!"
echo ""
echo "To run the container:"
echo "  docker-compose up -d eomt"
echo "  docker-compose exec eomt bash"
echo ""
echo "Or directly:"
echo "  docker run --gpus all -it --rm \\"
echo "    -v \$(pwd)/data:/workspace/data:ro \\"
echo "    -v \$(pwd)/checkpoints:/workspace/checkpoints \\"
echo "    -v \$(pwd)/wandb:/workspace/eomt/wandb \\"
echo "    eomt:latest"
echo ""
echo "For Jupyter notebook:"
echo "  docker-compose up -d jupyter"
echo "  # Access at http://localhost:8889"
echo ""
echo "Don't forget to:"
echo "1. Place your dataset zip files in ./data/"
echo "2. Set WANDB_API_KEY if using Weights & Biases"
echo "3. Run 'wandb login' inside the container"
