# Docker Setup for EoMT

This directory contains Docker configuration files to run EoMT in a containerized environment.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker runtime for GPU support
- CUDA-compatible GPU (recommended)

### Build and Run

1. **Build the Docker image:**
   ```bash
   docker build -t eomt:latest .
   ```

2. **Run with Docker Compose (recommended):**
   ```bash
   # For interactive development
   docker-compose up -d eomt
   docker-compose exec eomt bash
   
   # For Jupyter notebook
   docker-compose up -d jupyter
   # Access at http://localhost:8889
   ```

3. **Run directly with Docker:**
   ```bash
   docker run --gpus all -it --rm \
     -v $(pwd)/data:/workspace/data:ro \
     -v $(pwd)/checkpoints:/workspace/checkpoints \
     -v $(pwd)/wandb:/workspace/eomt/wandb \
     eomt:latest
   ```

## Directory Structure

Make sure you have the following directory structure:

```
eomt/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── data/                    # Mount your datasets here
│   ├── train2017.zip
│   ├── val2017.zip
│   ├── annotations_trainval2017.zip
│   └── panoptic_annotations_trainval2017.zip
├── checkpoints/             # Model checkpoints will be saved here
└── wandb/                   # WandB logs will be stored here
```

## Training Example

Once inside the container:

```bash
# Set up WandB (if you haven't already)
wandb login

# Train a model
python3 main.py fit \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /workspace/data
```

## Environment Variables

You can set these environment variables in a `.env` file:

```bash
# .env file
WANDB_API_KEY=your_wandb_api_key_here
CUDA_VISIBLE_DEVICES=0,1,2,3
```

## GPU Support

The Docker setup assumes you have NVIDIA Docker runtime installed. If you don't have GPU support, you can run the CPU-only version by removing the `--gpus all` flag and the deploy section in docker-compose.yml.

## Development

For development, you can mount the source code as a volume:

```bash
docker-compose run --rm eomt bash
```

This allows you to edit code on your host machine and see changes reflected immediately in the container.

## Troubleshooting

1. **CUDA/GPU issues**: Make sure nvidia-docker is installed and your GPU drivers are up to date.

2. **Permission issues**: If you encounter permission issues with mounted volumes, you may need to adjust file ownership:
   ```bash
   sudo chown -R $(id -u):$(id -g) ./checkpoints ./wandb
   ```

3. **Memory issues**: For large models, ensure your system has sufficient RAM and GPU memory.
