# Development Dockerfile - Source code mounted as volumes
# Use NVIDIA CUDA 13 runtime image
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install Python 3.11 and essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install non-PyTorch dependencies first from the default PyPI index
# Remove system blinker package to avoid conflicts
RUN apt-get remove -y python3-blinker || true

# Install basic dependencies first
RUN pip install --no-cache-dir \
    opencv-python-headless>=4.8.0 \
    ultralytics>=8.0.196 \
    numpy>=1.24.0 \
    python-osc>=1.8.0

# Install PyTorch with CUDA support from PyTorch index
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch>=2.0.0 \
    torchvision>=0.15.0

# Copy only static config file - source code will be mounted as volumes
COPY config.json .

# Create log directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for OSC
EXPOSE 8081

# Health check using OSC port
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(5); s.connect(('localhost', 8081)); s.close()" || exit 1

# Default command (daemon mode with GPU support)
CMD ["python", "main.py", "--daemon", "--config", "/app/config.json"]
