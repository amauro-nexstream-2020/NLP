# Docker image for Pure Transformer
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention (optional but recommended)
RUN pip install flash-attn --no-build-isolation || echo "Flash Attention not available"

# Copy source code
COPY pure_transformer /app/pure_transformer

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (can be overridden)
CMD ["python", "-m", "pure_transformer.run_pretrain", "--config", "a100_3day", "--model", "medium"]
