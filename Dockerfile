# EoMT (Encoder-only Mask Transformer) Docker Image
# Based on the official setup instructions from README.md

FROM nvidia/cuda:12.5.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    vim \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (following README instructions)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -ya

# Create conda environment with Python 3.13.2 (as specified in README)
RUN conda create -n eomt python==3.13.2 -y

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "eomt", "/bin/bash", "-c"]

# Set working directory
WORKDIR /workspace/eomt

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies (following README instructions)
RUN conda run -n eomt python3 -m pip install -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for data and checkpoints
RUN mkdir -p /workspace/data /workspace/checkpoints

# Set the conda environment as default

# writes the sourcing line to /root/.bashrc
RUN conda init bash
RUN echo "conda activate eomt" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=eomt

# Expose common ports (for Jupyter, TensorBoard, WandB if needed)
EXPOSE 8888 6006

# Set the default command to activate conda environment and start bash
CMD ["conda", "run", "-n", "eomt", "/bin/bash"]
