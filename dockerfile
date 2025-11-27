FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime




# Install system dependencies including docker CLI
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    apt-transport-https \
    ca-certificates \
    build-essential \
    git \
    wget \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your pipeline
COPY . /app/
WORKDIR /app/

# Default entrypoint
ENTRYPOINT ["python", "main.py"]
