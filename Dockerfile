# Stage 1: Builder Stage - Install dependencies including build tools and CUDA toolkit components
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, build essentials, git, and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    git \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    portaudio19-dev \
    python3-setuptools \
    python3.10-distutils \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchaudio==2.5.1+cu121 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install DeepSpeed
ENV DS_BUILD_TRANSFORMER=1
ENV DS_BUILD_CPU_ADAM=0
ENV DS_BUILD_FUSED_ADAM=0
ENV DS_BUILD_UTILS=0
ENV DS_BUILD_OPS=0

RUN echo "Building DeepSpeed with flags: DS_BUILD_TRANSFORMER=${DS_BUILD_TRANSFORMER}, DS_BUILD_CPU_ADAM=${DS_BUILD_CPU_ADAM}, DS_BUILD_FUSED_ADAM=${DS_BUILD_FUSED_ADAM}, DS_BUILD_UTILS=${DS_BUILD_UTILS}, DS_BUILD_OPS=${DS_BUILD_OPS}" && \
    pip install --no-cache-dir deepspeed \
    || (echo "DeepSpeed install failed. Check build logs above." && exit 1)

# Copy requirements file first to leverage Docker cache
COPY --chown=1001:1001 requirements.txt .

# Install remaining Python dependencies from requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    || (echo "pip install -r requirements.txt FAILED." && exit 1)

# Pin ctranslate2 to a compatible version
RUN pip install --no-cache-dir "ctranslate2<4.5.0"

# Copy the application src
COPY --chown=1001:1001 src/ ./src/

# --- Stage 2: Runtime Stage ---
# Base image still needs CUDA toolkit for PyTorch/DeepSpeed/etc in the app
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies for the APP + gosu
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libsndfile1 \
    ffmpeg \
    libportaudio2 \
    python3-setuptools \
    python3.10-distutils \
    ninja-build \
    build-essential \
    g++ \
    curl \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory for the application
WORKDIR /app/src

# Copy installed Python packages from the builder stage
RUN mkdir -p /usr/local/lib/python3.10/dist-packages
COPY --chown=1001:1001 --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy the application src from the builder stage
COPY --chown=1001:1001 --from=builder /app/src /app/src

# <<<--- Keep other model pre-downloads --->>>
# <<<--- Silero VAD Pre-download --->>>
RUN echo "Preloading Silero VAD model..." && \
    python3 <<EOF
import torch
import os
try:
    # Note: Downloads will happen as root here, ownership fixed later
    cache_dir = os.path.expanduser("~/.cache/torch") # Will resolve to /root/.cache/torch
    os.environ['TORCH_HOME'] = cache_dir
    print(f"Using TORCH_HOME: {cache_dir}")
    torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True
    )
    print("Silero VAD download successful.")
except Exception as e:
    print(f"Error downloading Silero VAD: {e}")
    exit(1)
EOF

# <<<--- faster-whisper Pre-download --->>>
ARG WHISPER_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN echo "Preloading faster_whisper model: ${WHISPER_MODEL}" && \
    # Note: Downloads happen as root, cache dir likely ~/.cache/huggingface or similar
    python3 -c "import os; print(f\"Downloading STT model: {os.getenv('WHISPER_MODEL')}\"); import faster_whisper; model = faster_whisper.WhisperModel(os.getenv('WHISPER_MODEL'), device='cpu'); print('Model download successful.')" \
    || (echo "Faster Whisper download failed" && exit 1)

# <<<--- SentenceFinishedClassification Pre-download --->>>
RUN echo "Preloading SentenceFinishedClassification model..." && \
    # Note: Downloads happen as root
    python3 -c "from transformers import AutoTokenizer; \
                from huggingface_hub import hf_hub_download; \
                HG_MODEL = 'livekit/turn-detector'; \
                MODEL_REVISION = 'v0.2.0-intl'; \
                ONNX_FILENAME = 'model_q8.onnx'; \
                print('Downloading tokenizer...'); \
                tokenizer = AutoTokenizer.from_pretrained( \
                    HG_MODEL, \
                    revision=MODEL_REVISION, \
                    local_files_only=False, \
                    truncation_side='left' \
                ); \
                print('Downloading ONNX classification model...'); \
                local_path = hf_hub_download( \
                    repo_id=HG_MODEL, \
                    filename=ONNX_FILENAME, \
                    subfolder='onnx', \
                    revision=MODEL_REVISION, \
                    local_files_only=False \
                ); \
                print('Downloading languages.json config...'); \
                config_fname = hf_hub_download( \
                    repo_id=HG_MODEL, \
                    filename='languages.json', \
                    revision=MODEL_REVISION, \
                    local_files_only=False \
                ); \
                print('Model downloads successful.')" \
    || (echo "Sentence Classifier download failed" && exit 1)


# Create a non-root user and group - DO NOT switch to it here
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 --create-home appuser

# Ensure directories are owned by appuser - This prepares the image layers correctly
# The entrypoint will handle runtime permissions for volumes/cache
RUN mkdir -p /home/appuser/.cache && \
    chown -R appuser:appgroup /app && \
    chown -R appuser:appgroup /home/appuser && \
    # Also chown the caches potentially populated by root during build
    if [ -d /root/.cache ]; then chown -R appuser:appgroup /root/.cache; fi

# Copy and set permissions for entrypoint script
COPY --chown=1001:1001 entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# --- REMOVED USER appuser --- The container will start as root.

# --- Keep ENV vars ---
ENV HOME=/home/appuser
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1
ENV MAX_AUDIO_QUEUE_SIZE=50
ENV LOG_LEVEL=INFO
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV RUNNING_IN_DOCKER=true
ENV DS_BUILD_OPS=1
ENV DS_BUILD_CPU_ADAM=0
ENV DS_BUILD_FUSED_ADAM=0
ENV DS_BUILD_UTILS=0
ENV DS_BUILD_TRANSFORMER=1
ENV HF_HOME=${HOME}/.cache/huggingface
ENV TORCH_HOME=${HOME}/.cache/torch

# Expose the port the FastAPI application runs on
EXPOSE 8000

# Set the entrypoint script - This runs as root
ENTRYPOINT ["/entrypoint.sh"]
# Define the default command - This is passed as "$@" to the entrypoint script
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
