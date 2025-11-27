# Dockerfile for Hybrid LLM Pretraining on A100
# Optimized for CUDA 12.1 + PyTorch 2.x with flash-attn and mamba-ssm

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# Note: flash-attn and mamba-ssm require specific CUDA setup
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.65.0 \
    tokenizers>=0.13.0 \
    sentencepiece>=0.1.99 \
    datasets>=2.12.0 \
    pandas>=2.0.0 \
    transformers>=4.30.0 \
    wandb>=0.15.0 \
    tensorboard>=2.13.0 \
    clearml>=1.14.0 \
    lightning>=2.2.0 \
    bitsandbytes>=0.43.0 \
    pyyaml>=6.0 \
    requests>=2.31.0 \
    scikit-learn

# Install flash-attn (requires compilation, may take a while)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install mamba-ssm
RUN pip install --no-cache-dir mamba-ssm --no-build-isolation

# Copy the project code
COPY . /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["python", "-m", "hybrid_llm.run_pretrain", "--model-size", "large", "--train-preset", "single_gpu"]
